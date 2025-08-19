import os
import sys
import shutil
import torch
import torch.distributed as dist
from transformers import AutoConfig
from colorama import Fore, Style


class SpecForgeLM():
    ### Speculative LM base methods ###
    @property
    def base_model(self):
        raise NotImplemented

    @property
    def draft_model(self):
        return self._draft_model

    def set_draft_model(self, model):
        self.config.speculative_decoding_draft_model = model.__class__.__name__
        self._draft_model = model
        self.on_draft_model_set()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        kwargs.pop('config')
        return cls.load_speculative_model(path, **kwargs)

    @classmethod
    def from_basemodel(cls, base_config, base_model_path=None,
                       *, AlgoClass=None, algo_kwargs={}, **kwargs):
        if AlgoClass is None:
            import specforge_het.models
            AlgoClass = eval('specforge_het.models.'
                + base_config.speculative_decoding_algorithm)
        if base_model_path is None:
            base_model_path = base_config.speculative_decoding_base_model_path

        module_spec = sys.modules[cls.__module__].__spec__.name.split('.')[-1]
        base_config.auto_map = {}
        base_config.auto_map['AutoModelForCausalLM'] = f'{module_spec}.{cls.__name__}'
        base_config.speculative_decoding_base_model_path = base_model_path
        base_config.speculative_decoding_algorithm = AlgoClass.__name__

        sub_classes = (*cls.__bases__[::-1], AlgoClass)
        ns = {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}
        derived_cls = type(cls.__name__, sub_classes, ns)
        model = derived_cls.from_pretrained(base_model_path, config=base_config, **kwargs)
        model.modeling_file = sys.modules[cls.__module__].__file__
        merge_kwargs = base_config.to_dict()
        merge_kwargs.update(algo_kwargs)
        AlgoClass.__init__(model, **merge_kwargs)
        return model

    def save_speculative_model(self, path, **kwargs):
        self.config.save_pretrained(path, **kwargs)
        self.draft_model.config.save_pretrained(f'{path}/draft_model', **kwargs)
        torch.save(self.draft_model.state_dict(), f'{path}/draft_model/states.pt')
        shutil.copy(self.modeling_file, path)

    @classmethod
    def load_speculative_model(cls, path, **kwargs):
        config = AutoConfig.from_pretrained(path)
        model = cls.from_basemodel(config, **kwargs)
        import specforge_het.models
        DrafterClass = eval('specforge_het.models.'
            + model.config.speculative_decoding_draft_model)
        draft_config = AutoConfig.from_pretrained(f'{path}/draft_model', trust_remote_code=True)
        draft_model = DrafterClass(draft_config, model.config)
        model.set_draft_model(draft_model)

        draft_state_dict = torch.load('./output/temp_save/draft_model/states.pt', map_location='cpu')
        draft_model.load_state_dict(draft_state_dict, strict=True)
        return model

    ### Training ###
    @property
    def is_eval_loop(self):
        return hasattr(self, 'trainer') and self.trainer.control.should_evaluate

    @staticmethod
    def gather_values(inp_dict):
        out_dict = dict()
        world = int(os.environ.get("WORLD_SIZE", 1))
        item = lambda x: x.detach().item() if isinstance(x, torch.Tensor) else x
        for key, val in inp_dict.items():
            gather_vals = [item(val) for _ in range(world)]
            if world > 1:
                dist.all_gather_object(gather_vals, val)
            out_dict[key] = [item(x) for x in gather_vals]
        return out_dict

    def reduce_values(self, inp_dict):
        from collections.abc import Iterable
        out_dict = dict()
        for key, vals in inp_dict.items():
            if key.startswith('_'): continue

            if key.startswith('max_'):
                reduced_val = max(vals) if isinstance(vals, Iterable) else vals
            elif key.startswith('min_'):
                reduced_val = min(vals) if isinstance(vals, Iterable) else vals
            # for evalulation loop, num_items_in_batch is local mini-batch,
            # thus each mini-batch loss is an unbiased metric, we need to
            # do averaging here.
            elif key.startswith('avg_') or self.is_eval_loop:
                numerator = sum([x for x in vals if x is not None])
                denominator  = sum([1 for x in vals if x is not None])
                reduced_val = numerator / denominator
            else:
                numerator = sum([x for x in vals if x is not None])
                accurate_reduce = self.training_configs.average_tokens_across_devices
                world = int(os.environ.get("WORLD_SIZE", 1))
                denominator = 1 if accurate_reduce else world
                reduced_val = numerator / denominator
            out_dict[key] = round(reduced_val, 3)
        return out_dict

    def training_log(self, global_step, metrics):
        key_prefix = 'eval/' if self.is_eval_loop else 'train/'
        log_dict = dict(global_step=global_step)
        for key, val in metrics.items():
            log_dict[key_prefix + key] = val
        if getattr(self, 'wandb', None):
            self.wandb.log(log_dict)
        print(log_dict)

    def training_collect_and_log(self, metrics, flush=False):
        # gather across all nodes (tensors will be converted to floats)
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        if metrics is not None:
            metrics['_rank'] = rank
            gather_metrics = self.gather_values(metrics)
        else:
            gather_metrics = {}

        if rank != 0:
            return

        # append to logs until do_logging at the master node
        for key, lst in gather_metrics.items():
            if key not in self.gather_metrics:
                self.gather_metrics[key] = []
            self.gather_metrics[key] += lst

        # reduce and log for the update of accumulated batches
        if flush and self.gather_metrics:
            logging_steps = self.training_configs.logging_steps
            do_logging = (self.update_step % logging_steps == 0)
            if (do_logging or self.is_eval_loop):
                print(f'\n{Fore.GREEN}[accumulated mini-batches]')
                for key, lst in self.gather_metrics.items():
                    if isinstance(lst[0], float):
                        print('\t', key, '=', [round(x, 3) for x in lst])
                    else:
                        print('\t', key, '=', lst)
                run_name = self.training_configs.run_name
                print(f'{Fore.YELLOW}[log] {run_name}', Style.RESET_ALL)
                reduced_metrics = self.reduce_values(self.gather_metrics)
                self.training_log(self.update_step, reduced_metrics)

            # clear the logs for accumulated batches
            self.gather_metrics = dict()

    def training_on_start(self, training_data_adapter):
        self.gather_metrics = dict()
        self.training_data_adapter = training_data_adapter
        self.forward = self.training_forward
        self.update_step = -1
        self.train()

    def auxiliary_training_process(self, forward_output, metrics):
        pass

    def training_forward(self, *args, **kwargs):
        if self.training_data_adapter:
            args, kwargs, data_metrics = self.training_data_adapter(*args, **kwargs)

        forward_output, metrics = self.speculative_forward(*args, **kwargs)
        self.auxiliary_training_process(forward_output, metrics)

        metrics = dict(**metrics, **data_metrics)
        if self.is_eval_loop:
            flush = False
        else:
            accelerator = (self.trainer.accelerator
                           if hasattr(self, 'trainer') else self.accelerator)
            flush = accelerator.sync_gradients
            self.update_step += (1 if flush else 0)
        #print('[backward]', forward_output['loss'].item())
        self.training_collect_and_log(metrics, flush=flush)

        return forward_output
