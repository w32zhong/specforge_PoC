from draco.models import *
from draco.speculative_decoding.base_hf import SpeculativeDecodingModelBaseHF


class VanillaSpeculativeDecodingModel(SpeculativeDecodingModelBaseHF):
    def configure(self, draft_depth=3, draft_beams=1, draft_topk=None, **kwargs):
        self.draft_depth = draft_depth
        self.draft_beams = draft_beams
        self.draft_topk = draft_depth if draft_topk is None else draft_topk

    @classmethod
    def from_composer(cls, target_model_config, draft_model_config, algorithm_config):
        model = super().from_composer(target_model_config, draft_model_config)
        model.configure(
            **model.configs_to_save(algorithm_config=algorithm_config.dict())
        )
        return model

