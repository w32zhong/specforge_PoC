from transformers import AutoConfig
from specforge_het.models import *
from sglang.srt.models.registry import ModelRegistry


for sgl_model in SGL_MODELS:
    ModelRegistry.models[sgl_model.__name__] = sgl_model


def adapted(model_path):
    base_model_config = AutoConfig.from_pretrained(model_path)
    base_model_path = base_model_config.speculative_decoding_base_model_path

    draft_model_path = f'{model_path}/draft_model'
    draft_model_config = AutoConfig.from_pretrained(draft_model_path)

    # inplace modification!
    draft_model_arch = base_model_config.speculative_decoding_draft_model + 'ForSGLang'
    if draft_model_arch not in draft_model_config.architectures:
        draft_model_config.architectures.insert(0, draft_model_arch)
        print('overwriting draft model config to support SGLang!')
        draft_model_config.save_pretrained(draft_model_path)

    return base_model_path, draft_model_path
