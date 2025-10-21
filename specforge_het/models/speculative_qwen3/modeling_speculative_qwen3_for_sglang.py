import torch
from functools import partial

from specforge_het.sglang_adapter_utils import SGLangAdapterMixin

from sglang.srt.layers.quantization.base_config import QuantizationConfig # resolve circ dep
from sglang.srt.models.qwen3 import *
from sglang.srt.utils import add_prefix


class FusedResidualIdentity(torch.nn.Identity):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return (hidden_states, None)
        else:
            return (hidden_states + residual, None)


def EagleV2_adapt(self, base_model_config, draft_model_config):
    if base_model_config.get('skip_first_input_layernorm', True):
        self.model.layers[0].input_layernorm = torch.nn.Identity()
        self.model.layers[0].layer_communicator.input_layernorm = torch.nn.Identity()
    if base_model_config.get('skip_output_norm', True):
        self.model.norm = FusedResidualIdentity()

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


class Qwen3DrafterForSGLang(Qwen3ForCausalLM, SGLangAdapterMixin):
    def __init__(self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""):

        super().__init__(config, quant_config, prefix)

        algo_cls, base_model_cfg, draft_model_cfg = self.bind_adapter(config)
        if algo_cls == 'EagleV2':
            EagleV2_adapt(self, base_model_cfg, draft_model_cfg)
        else:
            raise NotImplemented

    def load_weights(self, weights):
        weights = [(add_prefix(k, "model"), v) for k, v in weights]
        super().load_weights(self.warn_mismatch_weights(weights))

    # speculative decoding algorithm interfaces
    @property
    def draft_model(self):
        return self.model

    def get_hidden_size(self):
        return self.config.hidden_size
