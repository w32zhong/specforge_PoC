import torch


class EagleV2:
    def __init__(self, draft_layers=1, skip_input_layernorm=True, skip_output_norm=True, **kwargs):
        self.config.draft_layers = draft_layers
        self.config.skip_input_layernorm = skip_input_layernorm
        self.config.skip_output_norm = skip_output_norm

    def on_draft_model_set(self):
        base_hidden_size = self.get_hidden_size()
        draft_hidden_size = self.draft_model.get_hidden_size()
        self.draft_model.eagle_fc = torch.nn.Linear(
            2 * base_hidden_size, draft_hidden_size, bias=True
        )
        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

        if len(self.get_base_layers()) > 0:
            last_device = next(self.get_base_layers()[-1].parameters()).device
        else:
            last_device = self.device
        self.draft_model.to(last_device)

    @staticmethod
    def eagle_noise(tensor):
        seq_len = tensor.shape[1]
        return (torch.rand_like(tensor) - 0.5) * 0.2 * 512 / seq_len

    @staticmethod
    def eagle_data_offset(data):
        data['input_ids'] = data.pop('input_ids')[:, 1:]
        hidden_states = data.pop('encoder_outputs')
        data['encoder_outputs'] = hidden_states[:, :-1]
        data['target_hiddens'] = hidden_states[:, 1:].detach()
        data['labels'] = data.pop('labels')[:, 1:].detach()
        data['attention_mask'] = data['attention_mask'][:, 1:].detach()
        return data

    def speculative_forward(self, input_ids, encoder_outputs, target_hiddens, **kwargs):
        inputs_embeds = self.get_token_embedding(input_ids)
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        prev_encoder_hidden_states = encoder_outputs.to(device=device, dtype=dtype)
        next_encoder_hidden_states = target_hiddens.to(device=device, dtype=dtype)

        inputs_embeds = self.draft_model.eagle_fc(
            torch.cat((inputs_embeds, prev_encoder_hidden_states), dim=-1)
        )

        decoder_outputs = self.draft_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        predict = decoder_outputs[0]

        with torch.no_grad():
            target_logits = self.get_token_logits(next_encoder_hidden_states)
            target_p = torch.nn.Softmax(dim=2)(target_logits)
            pred_logits = self.get_token_logits(predict)

        labels = kwargs['labels']
        loss_mask = (labels != -100)
        loss_mask[:, -1] = 0
        loss_mask = loss_mask[:, :, None]

        pred_logp = torch.nn.LogSoftmax(dim=2)(pred_logits)
        plogp = target_p * pred_logp
        num_items_in_batch = kwargs.get('num_items_in_batch', loss_mask.sum())
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (num_items_in_batch + 1e-5)
        vloss = self.smooth_l1(predict, next_encoder_hidden_states)
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (num_items_in_batch + 1e-5)

        #loss = ploss + 10 * vloss # align with LM losses instead of (0.1 * ploss + vloss)
        loss = 0.1 * ploss + vloss
        return (
            dict(loss=loss, decoder_outputs=decoder_outputs, attention_mask=kwargs['attention_mask']),
            dict(loss=loss, ploss=ploss, vloss=vloss, _num_items_in_batch=num_items_in_batch)
        )
