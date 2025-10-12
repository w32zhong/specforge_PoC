import os
from transformers import AutoConfig
from specforge_het.models import *
from sglang.srt.models.registry import ModelRegistry
from huggingface_hub import snapshot_download

# Calling tqdm.tqdm.set_lock(threading.RLock()) pre-populates _lock
# with a plain thread lock, so that tqdm never uses the default
# multiprocessing RLock which would be tracked by multiprocessing/resource_tracker.py
# If we don't call this, you would get the “leaked semaphore” warning.
import tqdm, threading
tqdm.tqdm.set_lock(threading.RLock())


for sgl_model in SGL_MODELS:
    ModelRegistry.models[sgl_model.__name__] = sgl_model


def adapted(model_path):
    base_model_config = AutoConfig.from_pretrained(model_path)
    base_model_path = base_model_config.speculative_decoding_base_model_path

    try:
        model_path = snapshot_download(model_path)
    except:
        pass

    draft_model_path = os.path.join(model_path, 'draft_model')
    draft_model_config = AutoConfig.from_pretrained(draft_model_path)

    # any inplace modification?
    overwrite = False

    draft_model_arch = base_model_config.speculative_decoding_draft_model + 'ForSGLang'
    if draft_model_arch not in draft_model_config.architectures:
        draft_model_config.architectures.insert(0, draft_model_arch)
        overwrite = True

    base_key = '_base_model_config'
    if base_key not in draft_model_config:
        setattr(draft_model_config, base_key, base_model_config.to_json_string())
        overwrite = True

    if overwrite:
        print('overwriting draft model config to support SGLang!')
        draft_model_config.save_pretrained(draft_model_path)

    return base_model_path, draft_model_path
