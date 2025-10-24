import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import *
from compo import CompoConfigurable, CompoConfig


class Qwen3Drafter(Qwen3Model, CompoConfigurable):
    @classmethod
    def from_composer(cls, model_path=None, draft_hidden_size=None, draft_layers=1,
                      skip_first_input_layernorm=True, skip_output_norm=True, **kwargs):
        config = AutoConfig.from_pretrained(model_path)
        config.hidden_size = draft_hidden_size or config.hidden_size
        config.num_hidden_layers = draft_layers

        torch_dtype = eval(kwargs.get('torch_dtype', 'None'))
        device_map = kwargs.get('device_map', None)
        drafter = cls.from_pretrained(model_path,
                      config=config, torch_dtype=torch_dtype, device_map=device_map)

        if skip_first_input_layernorm:
            layer = drafter.layers[0]
            delattr(layer, 'input_layernorm')
            layer.input_layernorm = torch.nn.Identity()

        if skip_output_norm:
            delattr(drafter, 'norm')
            drafter.norm = torch.nn.Identity()

        delattr(drafter, 'embed_tokens')
        return drafter


class Qwen3ForTargetCausalLM(Qwen3ForCausalLM, CompoConfigurable):
    @property
    def base_model(self):
        return self.model

    @classmethod
    def from_composer(cls, model_path, **kwargs):
        torch_dtype = eval(kwargs.get('torch_dtype', 'None'))
        device_map = kwargs.get('device_map', None)
        return cls.from_pretrained(model_path,
                   config=None, torch_dtype=torch_dtype, device_map=device_map)


#class SpeculativeQwen3ForCausalLM(SpecForgeLM, Qwen3ForCausalLM):
#    @property
#    def base_model(self):
#        return self.model
#
#    def get_hidden_size(self):
#        return self.config.hidden_size
#
#    def get_base_layers(self):
#        return self.base_model.layers
#
#    def get_token_embedding(self, input_ids):
#        return self.base_model.embed_tokens(input_ids)
#
#    def get_positional_embedding(self, t, position_ids):
#        return self.base_model.rotary_emb(t, position_ids)
#
#    def get_token_logits(self, hidden_states):
#        return self.lm_head(hidden_states)
#
#    def get_max_ctx_length(self):
#        return self.model.config.max_position_embeddings
#
#    def save_pretrained(self, path, **kwargs):
#        return self.save_speculative_model(path, **kwargs)
