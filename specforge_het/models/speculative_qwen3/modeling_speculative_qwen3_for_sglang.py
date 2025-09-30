import json
import torch
from colorama import Fore, Style
from specforge_het.models import *
from sglang.srt.layers.quantization.base_config import QuantizationConfig # resolve circ dep
from sglang.srt.models.qwen3 import *
from sglang.srt.utils import add_prefix


class Qwen3DrafterForSGLang(Qwen3ForCausalLM):
    def __init__(self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""):

        # move and store base model config
        self._base_model_config_ = json.loads(config._base_model_config)
        delattr(config, '_base_model_config')

        # call default SGLang constructor
        super().__init__(config, quant_config, prefix)

        self.model.layers[0].input_layernorm = torch.nn.Identity()
        self.model.norm = torch.nn.Identity()
        del self.lm_head
        del self.model.embed_tokens

        # bind the algorithm
        speculative_algorithm = self._base_model_config_['speculative_decoding_algorithm']
        AlgoClass = eval(speculative_algorithm)
        AlgoClass.bind_model(self, load_device=None)

    # adaptor methods for EagleV2
    @property
    def draft_model(self):
        return self.model

    def get_hidden_size(self):
        return self.config.hidden_size

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
