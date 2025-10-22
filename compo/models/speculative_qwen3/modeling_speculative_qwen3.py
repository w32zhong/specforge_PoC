import torch
from transformers.models.qwen3.modeling_qwen3 import *
from specforge_het.specforge_lm import SpecForgeLM


class Qwen3Drafter(Qwen3Model):
    def __init__(self, draft_config, base_model):
        draft_config.num_hidden_layers = base_model.config.draft_layers
        draft_config.hidden_size = base_model.get_hidden_size()
        super().__init__(draft_config)

        if base_model.config.skip_first_input_layernorm:
            layer = self.layers[0]
            delattr(layer, 'input_layernorm')
            layer.input_layernorm = torch.nn.Identity()

        if base_model.config.skip_output_norm:
            delattr(self, 'norm')
            self.norm = torch.nn.Identity()

        delattr(self, 'embed_tokens')

    def get_hidden_size(self):
        return self.config.hidden_size


class SpeculativeQwen3ForCausalLM(SpecForgeLM, Qwen3ForCausalLM):
    @property
    def base_model(self):
        return self.model

    def get_hidden_size(self):
        return self.config.hidden_size

    def get_base_layers(self):
        return self.base_model.layers

    def get_token_embedding(self, input_ids):
        return self.base_model.embed_tokens(input_ids)

    def get_positional_embedding(self, t, position_ids):
        return self.base_model.rotary_emb(t, position_ids)

    def get_token_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def get_max_ctx_length(self):
        return self.model.config.max_position_embeddings

    def save_pretrained(self, path, **kwargs):
        return self.save_speculative_model(path, **kwargs)
