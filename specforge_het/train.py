import os
import uuid
import shutil
import torch
import random
import torch.distributed as dist
from functools import partial
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import HfArgumentParser
import transformers

from specforge_het.configs import Configs
from specforge_het.data_load import data_load
from specforge_het.model_load import load_models
from specforge_het.timer import TimeStats
from specforge_het.utils import *
from specforge_het.debug import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def direct_collator(tokenizer, max_length, batch):
    def convert_if_needed(data):
        try:
            data = torch.tensor(data)
        except (TypeError, ValueError):
            pass
        return data
    return {k: convert_if_needed([x[k] for x in batch]) for k in batch[0].keys()}


class MyCustomTrainer(Trainer):
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def _get_train_sampler(self, dataset):
        from torch.utils.data import SequentialSampler, RandomSampler
        if self.model.training_configs.sequential_loading:
            return SequentialSampler(self.train_dataset)
        else:
            return RandomSampler(self.train_dataset)

    def get_eval_dataloader(self, eval_dataset):
        dataloader_params = {
            "batch_size": 1,
            "collate_fn": self.eval_data_collator
        }
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare(eval_dataloader)

    def evaluate(self, **kwargs):
        eval_dataset = self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        for eval_idx, inputs in enumerate(dataloader):
            with torch.no_grad(), torch.cuda.amp.autocast():
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    master_print(dict(loss=outputs['loss'], eval_idx=eval_idx))

        model.training_collect_and_log(None, flush=True)

        # set flag False to indicate finishing eval loop early, because some eval-loop calls
        # like self.get_batch_samples() are invoked earlier than this flag is reset.
        self.control.should_evaluate = False


def train_eagle_pipeline(configs, run_name, tokenizer, model,
    train_dataset, eval_dataset, data_collator, eval_data_collator=None):

    from accelerate import Accelerator
    from transformers import get_constant_schedule_with_warmup
    from tqdm import tqdm
    import time

    tg = torch.Generator()
    tg.manual_seed(configs.seed)
    train_loader = DataLoader(train_dataset,
        batch_size=configs.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=1,
        generator=tg,
    )

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=configs.gradient_accumulation_steps)

    optimizer = torch.optim.AdamW(model.parameters(),
        lr=configs.learning_rate,
        betas=(configs.adam_beta1, configs.adam_beta2)
    )

    scheduler = get_constant_schedule_with_warmup(optimizer,
        num_warmup_steps=configs.warmup_steps)

    wandb = getattr(model, 'wandb', None)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if rank == 0:
        output_dir = configs.output_dir
        accelerator.save_model(model, f"{output_dir}/model_init")

    model.accelerator = accelerator
    print(model)

    for epoch in range(configs.num_train_epochs):
        print('epoch:', epoch)

        for batch_idx, data in enumerate(tqdm(train_loader)):

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                #cmp_tensor_w_another(data['input_ids'])
                #breakpoint()

                eagle_outputs = model(**data)
                loss = eagle_outputs['loss']

                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), configs.max_grad_norm)
                optimizer.step()
                scheduler.step()
                time.sleep(0.1)

        if rank == 0:
            output_dir = configs.output_dir
            accelerator.save_model(model, f"{output_dir}/model_{epoch}")


def train(configs, hgf_training_args, run_name, tokenizer, model,
    train_dataset, train_collator, eval_dataset, eval_collator):
    if not configs.overwrite_output_dir:
        assert not os.path.exists(configs.output_dir), configs.output_dir

    torch.autograd.set_detect_anomaly(configs.debug)

    if configs.debug:
        torch.set_printoptions(precision=2, sci_mode=False)
        transformers.logging.set_verbosity_info()

    if torch.__version__.startswith('2.2'):
        model.to(f'cuda:{rank}')

    def EAGLE_training_data_adapter(*args, **kwargs):
        kwargs['output_router_logits'] = True

        if 'target_hiddens' not in kwargs: # raw data, no offset yet
            data = model.eagle_data_offset(data=kwargs)
            data['encoder_outputs'] += model.eagle_noise(data['encoder_outputs'])
        else:
            data = kwargs

        labels = data['labels']
        metrics = dict(
            _ids = data['data_ids'],
            max_seq_len = (labels != -100).sum(-1).max(),
            min_seq_len = (labels != -100).sum(-1).min(),
        )
        return args, kwargs, metrics

    model.training_on_start(EAGLE_training_data_adapter)

    if configs.use_eagle_pipeline:
        return train_eagle_pipeline(configs, run_name, tokenizer, model,
            train_dataset, eval_dataset,
            partial(train_collator, tokenizer, configs.max_length)
        )

    master_print('[train_dataset]', train_dataset)
    master_print('[eval_dataset]', eval_dataset)
    collator = lambda c: partial(c, tokenizer, configs.max_length) if c else None
    trainer = MyCustomTrainer(
        model=model,
        args=hgf_training_args,
        train_dataset=train_dataset,
        data_collator=collator(train_collator),
        eval_dataset=eval_dataset,
        eval_data_collator=collator(eval_collator),
        compute_metrics=None # we use our customized one
    )

    def default_getter(batch_samples):
        return sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
    if configs.use_default_num_items_getter:
        trainer.get_num_items_in_batch = default_getter
    else:
        trainer.get_num_items_in_batch = model.get_num_items_in_batch

    model.trainer = trainer
    model.timer = TimeStats(disable=(not configs.debug))
    trainer.train(resume_from_checkpoint=configs.resume_from_checkpoint)


def extract_training_args(run_name, configs):
    training_args = dict(
        run_name=run_name,
        remove_unused_columns=False,
        output_dir=configs.output_dir,
        overwrite_output_dir=configs.overwrite_output_dir,
        save_strategy=configs.save_strategy,
        save_steps=configs.save_steps,
        save_total_limit=configs.save_total_limit,
        optim=configs.optim,
        eval_strategy=configs.eval_strategy,
        eval_steps=configs.eval_steps,
        per_device_eval_batch_size=configs.per_device_eval_batch_size,
        num_train_epochs=configs.num_train_epochs,
        max_steps=configs.max_steps,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        per_device_train_batch_size=configs.per_device_train_batch_size,
        average_tokens_across_devices=configs.average_tokens_across_devices,
        warmup_steps=configs.warmup_steps,
        lr_scheduler_type=configs.lr_scheduler_type,
        learning_rate=configs.learning_rate,
        ddp_find_unused_parameters=configs.ddp_find_unused_parameters,
        max_grad_norm=configs.max_grad_norm,
        adam_beta1=configs.adam_beta1,
        adam_beta2=configs.adam_beta2,
        ignore_data_skip=False, # otherwise epoch will reset when resume
        logging_nan_inf_filter=False,
        logging_steps=configs.logging_steps,
        logging_first_step=configs.logging_first_step,
        bf16=configs.bf16,
        tf32=configs.tf32,
        report_to=configs.report_to,
        deepspeed=configs.deepspeed,
        ddp_backend=configs.ddp_backend,
        dataloader_drop_last=configs.dataloader_drop_last,
        eval_on_start=(configs.deepspeed is not None), # to load ckpt to ds_engine
        batch_eval_metrics=True
    )

    if configs.deepspeed:
        parser = HfArgumentParser(TrainingArguments)
        def dict_to_args_list(config_dict):
            args_list = []
            for key, value in config_dict.items():
                args_list.extend([f"--{key}", str(value)])
            return args_list
        training_args = dict_to_args_list(training_args)
        training_args, unused_args = parser.parse_args_into_dataclasses(
            return_remaining_strings=True,
            args=training_args
        )
        assert len(unused_args) == 0
    else:
        training_args = TrainingArguments(**training_args)

    return training_args


def main(config_file='configs.ini', **injects):
    assert torch.cuda.is_available()
    configs = Configs.from_config_file(config_file, **injects)

    from accelerate.utils import set_seed
    set_seed(configs.training.seed)
    #torch.use_deterministic_algorithms(True)

    # record world size
    print('[rank]', rank, '/', world_size)
    configs.set_obj('training.world_size', world_size)

    # record device info (using temp init_process_group)
    launcher_pid = os.getppid()
    backend = configs.training.ddp_backend
    dist.init_process_group(backend, rank=rank, world_size=world_size,
        init_method=f"file:///tmp/torch_temp_dist-{launcher_pid}")
    device_name = torch.cuda.get_device_name(rank)
    gather_device_names = [device_name for _ in range(world_size)]
    torch.cuda.set_device(rank)
    if world_size > 1:
        dist.all_gather_object(gather_device_names, device_name)
    master_print('[gpus]', gather_device_names)
    configs.set_obj('device_names', gather_device_names)

    # record git info
    try:
        import git
        git_repo = git.Repo(search_parent_directories=True)
        git_sha1 = git_repo.head.object.hexsha
        git_diff = git_repo.git.diff()
        configs.set_obj('training.git_sha1', git_sha1)
        configs.set_obj('training.git_diff', git_diff)
        master_print('[git]', git_sha1)
    except:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # use an input-insensitive scaled-dot-product kernel?
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_mem_efficient=True, enable_math=True
    )

    wandb = None
    if rank == 0:
        if configs.training.report_to == 'wandb':
            # let wandb assigns run_name
            import wandb
            wandb.init(
                project=configs.training.project,
                config=configs.get_obj(),
                id=configs.training.resume_wandb_runid
            )
            if configs.training.resume_wandb_runid is None:
                run_name = wandb.run.name
            else:
                # use explicitly configured run_name
                run_name = configs.training.run_name
        else:
            # use default run_name
            run_name = configs.training.run_name
    else:
        run_name = None

    # sync run_name
    world = int(os.environ.get("WORLD_SIZE", 1))
    gather_names = [run_name for _ in range(world)]
    if world > 1: dist.all_gather_object(gather_names, run_name)
    run_name = gather_names[0] # always use rank-1 node run_name
    configs.set_obj('training.run_name', run_name)
    print(f'[rank#{rank} run name]', run_name)

    # record output dir and log finalized configs before training
    output_dir = os.path.join(configs.training.output_dir, run_name)
    configs.set_obj('training.output_dir', output_dir)
    configs.save_json(output_dir, fname='specforge_het.json')
    master_print(configs)

    # potentially setup deepspeed, should be preceding load_models()
    hgf_training_args = extract_training_args(run_name, configs.training)
    dist.destroy_process_group()

    tokenizer, model = load_models(
        configs.modeling, world_size, rank,
        use_deepspeed=configs.training.deepspeed
    )

    if configs.training.model_init_ckpt:
        model_init_dict = torch.load(configs.training.model_init_ckpt, weights_only=False)
        converted_dict = dict()
        for key, val in model_init_dict.items():
            if key.startswith('eagle_fc.'):
                converted_dict[key.replace('eagle_fc.', '_draft_model.eagle_fc.')] = val
            elif key.startswith('speculative_decoder.'):
                converted_dict[key.replace('speculative_decoder.', '_draft_model.')] = val
            else:
                converted_dict[key] = val
        model.load_state_dict(converted_dict, strict=True)

    datasets_and_collators = data_load(configs.dataset)

    model.wandb = wandb
    model.tokenizer = tokenizer # debug-only
    model.training_configs = configs.training

    train(configs.training, hgf_training_args,
          run_name, tokenizer, model, *datasets_and_collators)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
