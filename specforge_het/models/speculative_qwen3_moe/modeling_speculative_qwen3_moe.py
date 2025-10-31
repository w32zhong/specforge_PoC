import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import *
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from specforge_het.specforge_lm import SpecForgeLM
from specforge_het.utils import recycle_vram
from functools import partial


class Qwen3MoeDrafter(Qwen3MoeModel):
    def __init__(self, draft_config, base_model):
        draft_config.num_hidden_layers = base_model.config.draft_layers
        draft_config.hidden_size = base_model.get_hidden_size()

        super().__init__(draft_config)

        if getattr(draft_config, 'zero_compute_experts', None):
            for layer in self.layers:
                for _ in range(draft_config.zero_compute_experts):
                    layer.mlp.experts.pop(-1)
                    layer.mlp.experts.append(torch.nn.Identity())
                recycle_vram()

        if getattr(draft_config, 'shared_experts', None):
            intermediate_size = draft_config.moe_intermediate_size * draft_config.shared_experts
            for layer in self.layers:
                layer.mlp.shared_experts = Qwen3MoeMLP(
                        draft_config, intermediate_size=intermediate_size)
                layer.mlp.forward = partial(self.shared_expert_forward_wrap,
                        layer.mlp, origin_forward=layer.mlp.forward)

        if base_model.config.skip_first_input_layernorm:
            layer = self.layers[0]
            delattr(layer, 'input_layernorm')
            layer.input_layernorm = torch.nn.Identity()

        if base_model.config.skip_output_norm:
            delattr(self, 'norm')
            self.norm = torch.nn.Identity()

        delattr(self, 'embed_tokens')

    @staticmethod
    def shared_expert_forward_wrap(self, hidden_states, origin_forward=None):
        # shared experts
        residuals = self.shared_experts(hidden_states)
        # routed experts
        hidden_states, router_logits = origin_forward(hidden_states)
        # add outputs
        return hidden_states + residuals, router_logits

    def get_hidden_size(self):
        return self.config.hidden_size

    def auxiliary_training_process(self, forward_output, metrics):
        loss = forward_output.pop('loss')
        router_logits = forward_output['decoder_outputs'].router_logits
        lb_loss = load_balancing_loss_func(
            router_logits,
            self.config.num_experts,
            self.config.num_experts_per_tok,
            forward_output['attention_mask']
        ).to(loss.device)
        metrics['lb_loss'] = lb_loss
        loss += self.config.router_aux_loss_coef * lb_loss

        forward_output['loss'] = loss
        metrics['loss'] = loss


class SpeculativeQwen3MoeForCausalLM(SpecForgeLM, Qwen3MoeForCausalLM):
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
