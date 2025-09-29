import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig # resolve circ dep
from sglang.srt.models.qwen3 import *
from sglang.srt.utils import add_prefix

class Qwen3DrafterForSGLang(Qwen3ForCausalLM):
    def __init__(self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""):
        super().__init__(config, quant_config, prefix)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

    def load_weights(self, weights):
        import rpdb; rpdb.set_trace()

    #def __init__(self, draft_config, base_model):
    #    draft_config.num_hidden_layers = base_model.config.draft_layers
    #    draft_config.hidden_size = base_model.get_hidden_size()
    #    super().__init__(draft_config)

    #    if base_model.config.skip_first_input_layernorm:
    #        layer = self.layers[0]
    #        delattr(layer, 'input_layernorm')
    #        layer.input_layernorm = torch.nn.Identity()

    #    if base_model.config.skip_output_norm:
    #        delattr(self, 'norm')
    #        self.norm = torch.nn.Identity()

    #    delattr(self, 'embed_tokens')

    #def get_hidden_size(self):
    #    return self.config.hidden_size
