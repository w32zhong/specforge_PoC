from draco.models import *
from draco.speculative_decoding.base_hf import SpeculativeDecodingModelBaseHF


class VanillaSpeculativeDecodingModel(SpeculativeDecodingModelBaseHF):

    def configure(self, draft_depth=3, draft_beams=1, draft_topk=None, **kwargs):
        self.draft_depth = draft_depth
        self.draft_beams = draft_beams
        self.draft_topk = draft_depth if draft_topk is None else draft_topk

    @classmethod
    def from_composer(cls,
                      target_model_config=None,
                      draft_model_config=None,
                      algorithm_config=None, **kwargs):
        model = super().from_composer(
            target_model_config=target_model_config,
            draft_model_config=draft_model_config
        )
        model.configure(**algorithm_config.dict())
        return model


if __name__ == '__main__':
    from draco import CompoConfig
    cfg = CompoConfig.from_composer('./configs_v2.ini')
    model = VanillaSpeculativeDecodingModel.from_composer_config(cfg.speculative_decoding)
    model.save_pretrained('./output/temp_save')
    breakpoint()
