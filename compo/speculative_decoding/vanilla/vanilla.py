from compo import CompoConfig
from compo.models import *
from compo.speculative_decoding.base import SpeculativeDecodingModelBase


class VanillaSpeculativeDecodingModel(SpeculativeDecodingModelBase):

    @classmethod
    def from_composer(cls, draft_model=None, target_model=None, algorithm=None, **kwargs):

        speculative_model = super().from_composer(
            draft_model_name=draft_model.class_name, target_model=target_model
        )

        DraftModel = eval(draft_model.class_name)
        draft_model = DraftModel.from_composer_config(draft_model)

        speculative_model.set_draft_model(draft_model)
        return speculative_model


if __name__ == '__main__':
    cfg = CompoConfig.from_composer('./configs_v2.ini')
    model = VanillaSpeculativeDecodingModel.from_composer_config(cfg.speculative_decoding)
    breakpoint()
