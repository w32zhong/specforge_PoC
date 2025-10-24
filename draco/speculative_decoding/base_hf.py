from draco import CompoConfigurable, CompoConfig
from draco.models import *


class SpeculativeDecodingModelBaseHF(CompoConfigurable):
    _speculative_config_prefix = '_speculative_decoding_configs'
    _draft_model_path_prefix = '_draft_model'

    @property
    def draft_model(self):
        return getattr(self, self._draft_model_path_prefix)

    def set_draft_model(self, draft_model, draft_model_config):
        draft_device = CompoConfig.resolve(draft_model_config.device)
        if draft_device == 'auto':
            # send to the last device (usually the same device of the last layer)
            max_device_ordinal = max(self.hf_device_map.values())
            draft_model.to(max_device_ordinal)
        elif draft_device:
            draft_model.to(draft_device)
        else:
            draft_model.to(self.base_model.device)

        setattr(self, self._draft_model_path_prefix, draft_model)

    @classmethod
    def dynamic_typed_base_model(cls, target_model, draft_model_name='UnknownDrafter'):
        TargetModel = eval(target_model.class_name)
        SpeculativeModel = type(
            f'{cls.__name__}.{TargetModel.__name__}.{draft_model_name}',
            (cls, TargetModel),
            {
                'from_composer': classmethod(TargetModel.from_composer.__func__),
                'from_composer_config': classmethod(TargetModel.from_composer_config.__func__)
            }
        )
        return SpeculativeModel.from_composer_config(target_model)

    @classmethod
    def from_composer(cls, target_model_config=None, draft_model_config=None, **kwargs):
        # bind to target model as the base model
        base_model = cls.dynamic_typed_base_model(target_model_config, draft_model_config.class_name)

        # create draft model
        if draft_model_config is not None:
            DraftModel = eval(draft_model_config.class_name)
            draft_model = DraftModel.from_composer_config(draft_model_config)
            base_model.set_draft_model(draft_model, draft_model_config)

        # save runtime config
        setattr(base_model, cls._speculative_config_prefix, dict(
            target_model_config=target_model_config.dict(),
            draft_model_config=draft_model_config.dict()
        ))

        return base_model

    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        model = super().from_pretrained(
            path, config=config, **kwargs
        )
        return model

    def save_pretrained(self, path, **kwargs):
        config_dict = getattr(self, self._speculative_config_prefix, {})
        config = CompoConfig(config_dict)
        config.save_json_file(path)

        model = self.draft_model.save_pretrained(
            path, **kwargs
        )
        return model
