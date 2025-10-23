from abc import abstractmethod
from compo import CompoConfigurable
from compo.models import *


class SpeculativeDecodingModelBase(CompoConfigurable):
    _draft_model_path_prefix = '_draft_model'

    @property
    def draft_model(self):
        return getattr(self, self._draft_model_path_prefix)

    def set_draft_model(self, model):
        setattr(self, self._draft_model_path_prefix, model)

    @classmethod
    def from_composer(cls, draft_model_name=None, target_model=None, **kwargs):

        TargetModel = eval(target_model.class_name)

        SpeculativeModel = type(
            f'{cls.__name__}.{TargetModel.__name__}.{draft_model_name}',
            (cls, TargetModel),
            {
                'from_composer': classmethod(TargetModel.from_composer.__func__),
                'from_composer_config': classmethod(TargetModel.from_composer_config.__func__)
            }
        )

        speculative_model = SpeculativeModel.from_composer_config(target_model)
        return speculative_model

    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        model = super().from_pretrained(
            model_path, config=config, **kwargs
        )
        return model
