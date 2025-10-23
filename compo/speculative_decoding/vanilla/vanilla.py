from abc import abstractmethod
from compo import CompoConfigurable, CompoConfig
from compo.models import *


class VanillaSpeculativeDecodingModel(CompoConfigurable):
    _draft_model_path_prefix = '_draft_model'

    @property
    @abstractmethod
    def base_model(self): ...

    @property
    def draft_model(self):
        return getattr(self, self._draft_model_path_prefix)

    @classmethod
    def from_composer(cls, draft_model=None, target_model=None, **kwargs):
        DraftModel = eval(draft_model.class_name)
        TargetModel = eval(target_model.class_name)

        SpeculativeModel = type(
            f'{cls.__name__}.{TargetModel.__name__}.{DraftModel.__name__}',
            (TargetModel, cls),
            {}
        )

        speculative_model = SpeculativeModel.from_composer_config(target_model)
        draft_model = DraftModel.from_composer_config(draft_model)
        speculative_model.draft_model = draft_model
        return speculative_model


if __name__ == '__main__':
    cfg = CompoConfig.from_composer('./configs_v2.ini')
    model = VanillaSpeculativeDecodingModel.from_composer_config(cfg.speculative_decoding)
