SGL_MODELS = []

from .algorithms.eagle_v2 import EagleV2

from .speculative_llama.modeling_speculative_llama import SpeculativeLlamaForCausalLM, LlamaDrafter

from .speculative_qwen3.modeling_speculative_qwen3 import SpeculativeQwen3ForCausalLM, Qwen3Drafter
try:
    from .speculative_qwen3.modeling_speculative_qwen3_for_sglang import Qwen3DrafterForSGLang
    SGL_MODELS += [Qwen3DrafterForSGLang]
except ImportError:
    pass

from .speculative_qwen3_moe.modeling_speculative_qwen3_moe import SpeculativeQwen3MoeForCausalLM, Qwen3MoeDrafter
