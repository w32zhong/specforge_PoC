from compo import CompoConfigurable, CompoConfig


class SpeculativeDecodingModel(CompoConfigurable):
    draft_model_path_prefix = '_draft_model'

    @property
    @abstractmethod
    def base_model(self): ...

    @property
    def draft_model(self):
        return getattr(self, self.draft_model_path_prefix)

    @classmethod
    def from_composer(cls, base_model_configs=None, draft_model_configs=None, speculative_algo_configs=None):

        SpeculativeModel = type(
            f'Speculative{BaseModel.__name__}And{DraftModel.__name__}Using{SpeculativeAlgo.__name__}',
            (BaseModel, cls, SpeculativeAlgo),
            {"from_composer": classmethod(BaseModel.from_composer.__func__)}
        )

        model = SpeculativeModel.from_composer(**)

        drafter = draft_cls.from_composer(**draft_model_configs)
        model.draft_model = drafter
        return model
