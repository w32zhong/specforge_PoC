import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import *
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from specforge_het.specforge_lm import SpecForgeLM


class Qwen3MoeDrafter(Qwen3MoeModel):
    def __init__(self, draft_config, base_model):
        draft_config.num_hidden_layers = base_model.config.draft_layers
        draft_config.hidden_size = base_model.get_hidden_size()
        super().__init__(draft_config)

        if base_model.config.skip_input_layernorm:
            for layer in self.layers:
                delattr(layer, 'input_layernorm')
                layer.input_layernorm = torch.nn.Identity()

        if base_model.config.skip_output_norm:
            delattr(self, 'norm')
            self.norm = torch.nn.Identity()

        delattr(self, 'embed_tokens')

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

    def get_token_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def save_pretrained(self, path, **kwargs):
        return self.save_speculative_model(path, **kwargs)
