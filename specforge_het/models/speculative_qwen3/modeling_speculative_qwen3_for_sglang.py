import json
import torch
from functools import partial
from colorama import Fore, Style
from specforge_het.models import *
from sglang.srt.layers.quantization.base_config import QuantizationConfig # resolve circ dep
from sglang.srt.models.qwen3 import *
from sglang.srt.utils import add_prefix


class ReturnTupleIdentity(torch.nn.Identity):
    def forward(self, *args, **kwargs):
        input = args[0] if args else next(iter(kwargs.values()))
        return (input, None)


def EagleV2_adapt(self, base_model_config, draft_model_config):
    if base_model_config.get('skip_first_input_layernorm', True):
        del self.model.layers[0].input_layernorm
    if base_model_config.get('skip_output_norm', True):
        self.model.norm = ReturnTupleIdentity()

    # Embedding layers are handled in SGLang Worker.set_embed_and_head().
    # Under tie_word_embeddings, this could be deleted twice by the SGLang
    # speculative decoding worker. So let's create a placeholder here.
    if draft_model_config.tie_word_embeddings:
        assert id(self.lm_head.weight) == id(self.model.embed_tokens.weight)
        self.lm_head = torch.nn.Linear(1,1)

    def fwd_hook(self, input_ids, positions, forward_batch, input_embeds, **kwargs):
        assert self.pp_group.is_first_rank
        origin_forward = kwargs.pop('origin_forward')
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        prev_states = forward_batch.spec_info.hidden_states
        hidden_states = self.eagle_fc(
            torch.cat((hidden_states, prev_states), dim=-1)
        )
        return origin_forward(input_ids, positions, forward_batch,
                              hidden_states, kwargs['pp_proxy_tensors'])
    self.model.forward = partial(fwd_hook, self.model,
                                 origin_forward=self.model.forward)



class Qwen3DrafterForSGLang(Qwen3ForCausalLM):
    def __init__(self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""):

        # move and store base model config
        base_model_config = json.loads(config._base_model_config)
        delattr(config, '_base_model_config')

        # call default SGLang constructor
        super().__init__(config, quant_config, prefix)

        # bind the algorithm
        speculative_algorithm = base_model_config['speculative_decoding_algorithm']
        AlgoClass = eval(speculative_algorithm)
        AlgoClass.bind_model(self, load_device=None)

        if AlgoClass.__name__ == 'EagleV2':
            EagleV2_adapt(self, base_model_config, config)
        else:
            raise NotImplementedError

    # overriding SGLang methods
    def load_weights(self, weights):
        weights = [(add_prefix(k, "model"), v) for k, v in weights]

        model_params_keys = set([key for key, _ in self.named_parameters()])
        load_weights_keys = set([key for key, _ in weights])

        expect_extra_keys = model_params_keys - load_weights_keys
        loading_extra_keys = load_weights_keys - model_params_keys

        if expect_extra_keys or loading_extra_keys:
            print(
                f'{Fore.YELLOW}Loading weight keys mismatch: \n\n' +
                f'model_params_keys: {model_params_keys}\n\n' +
                f'expect_extra_keys: {expect_extra_keys}\n\n' +
                f'loading_extra_keys: {loading_extra_keys}\n\n' +
                Style.RESET_ALL
            )
            #import rpdb; rpdb.set_trace()

        super().load_weights(weights)

    # speculative decoding algorithm interfaces
    @property
    def draft_model(self):
        return self.model

    def get_hidden_size(self):
        return self.config.hidden_size
