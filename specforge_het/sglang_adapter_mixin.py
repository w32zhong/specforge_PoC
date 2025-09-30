import json
from colorama import Fore, Style
from specforge_het.models import *


class SGLangAdapterMixin:
    def bind_adapter(self, config):
        # move and store base model config
        base_model_config = json.loads(config._base_model_config)
        delattr(config, '_base_model_config')

        # bind the algorithm
        speculative_algorithm = base_model_config['speculative_decoding_algorithm']
        AlgoClass = eval(speculative_algorithm)
        AlgoClass.bind_model(self, load_device=None)

        return AlgoClass.__name__, base_model_config, config

    def warn_unmatch_weights(self, weights):
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

        return weights
