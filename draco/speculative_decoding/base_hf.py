import os
import logging
from typing import runtime_checkable, Protocol
from draco.config import CompoConfigurable, CompoConfig
from draco.models import *

logger = logging.getLogger(__name__)


@runtime_checkable
class TargetModelProtocol(Protocol):
    @property
    def target_model(self): ...


class SpeculativeDecodingModelBaseHF(CompoConfigurable):
    _speculative_config_prefix = '_speculative_decoding_configs'
    _draft_model_attr_prefix = '_draft_model'
    _draft_model_save_subdir = 'draft_model'
    _save_conf_fname = 'speculative_decoding.json'

    @property
    def draft_model(self):
        return getattr(self, self._draft_model_attr_prefix)

    def set_draft_model(self, draft_model, draft_model_config):
        draft_device = CompoConfig.resolve(draft_model_config.device)
        if draft_device == 'auto':
            # send to the last device (usually the same device of the last layer)
            max_device_ordinal = max(self.hf_device_map.values())
            draft_model.to(max_device_ordinal)
        elif draft_device:
            draft_model.to(draft_device)
        else:
            draft_model.to(self.target_model.device)

        setattr(self, self._draft_model_attr_prefix, draft_model)

    @classmethod
    def dynamic_typed_base_model(cls, target_model_config, draft_model_name='UnknownDrafter'):
        TargetModel = eval(target_model_config.class_name)
        assert isinstance(TargetModel, TargetModelProtocol), (
            f'{TargetModel} incompatible as a target model!'
        )

        SpeculativeModel = type(
            f'{cls.__name__}.{TargetModel.__name__}.{draft_model_name}',
            (cls, TargetModel),
            {
                'from_composer': classmethod(TargetModel.from_composer.__func__),
                'from_composer_config': classmethod(TargetModel.from_composer_config.__func__)
            }
        )
        return SpeculativeModel.from_composer_config(target_model_config)

    @classmethod
    def from_composer(cls, target_model_config, draft_model_config):
        # bind to target model as the base model
        base_model = cls.dynamic_typed_base_model(target_model_config, draft_model_config.class_name)

        # create draft model
        if draft_model_config is not None:
            DraftModel = eval(draft_model_config.class_name)
            draft_model = DraftModel.from_composer_config(draft_model_config)
            base_model.set_draft_model(draft_model, draft_model_config)

        # save runtime config for later save_pretrained()
        base_model.configs_to_save(
            target_model_config=target_model_config.dict(),
            draft_model_config=draft_model_config.dict()
        )

        return base_model

    def configs_to_save(self, **kwargs):
        if d := getattr(self, self._speculative_config_prefix, {}):
            d.update(kwargs)
        else:
            setattr(self, self._speculative_config_prefix, kwargs)
        return d

    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        # should be saved to these places if they are "save_pretrained" from this class.
        config_file = os.path.join(path, cls._save_conf_fname)
        draft_model_save_dir = os.path.join(path, cls._draft_model_save_subdir)

        if os.path.exists(config_file):
            configs = CompoConfig({})
            configs.load_json_file(config_file, warn_change_key_prefix=None)
            configs.draft_model_config.model_path = draft_model_save_dir
            return cls.from_composer_config(configs)
        else:
            model = super().from_pretrained(
                path, config=config, **kwargs
            )
            return model

    def save_pretrained(self, path, **kwargs):
        # load runtime config
        config = CompoConfig(self.configs_to_save())

        # save draft model
        draft_model_save_dir = os.path.join(path, self._draft_model_save_subdir)
        logger.info(f'Saving draft model to {draft_model_save_dir} ...')
        self.draft_model.save_pretrained(draft_model_save_dir, **kwargs)

        if draft_model_path := config.draft_model_config.model_path:
            config.draft_model_config.origin_model_path = draft_model_path
            config.draft_model_config.model_path = None

        # save target model (only if specified because target model is usually large)
        if config.target_model_config.save_model:
            save_draft_model = self.draft_model
            setattr(self, self._draft_model_attr_prefix, None)
            logger.info(f'Saving target model to {path} ...')
            super().save_pretrained(path, **kwargs)
            setattr(self, self._draft_model_attr_prefix, save_draft_model)

            if target_model_path := config.target_model_config.model_path:
                config.target_model_config.origin_model_path = target_model_path
                config.target_model_config.model_path = None

        # save composer config
        config.save_json_file(path, fname=self._save_conf_fname)
