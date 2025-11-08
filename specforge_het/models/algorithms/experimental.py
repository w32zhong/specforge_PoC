import torch, random
from specforge_het.models.algorithms.eagle_v2 import EagleV2, print_predictions, prediction_accuracy

class Experimental(EagleV2):

    @staticmethod
    def configure(config, draft_layers=1,
                  skip_first_input_layernorm=True,
                  skip_output_norm=False,
                  draft_hidden_size=1408,
                  draft_intermediate_size=5376,
                  latent_initializer='random',
                  H_freq=1,
                  L_freq=3,
                  **kwargs):
        super(Experimental, Experimental).configure(config,
                          draft_layers=draft_layers,
                          skip_first_input_layernorm=skip_first_input_layernorm,
                          skip_output_norm=skip_output_norm,
                          **kwargs)

        config.draft_hidden_size = draft_hidden_size
        config.draft_intermediate_size = draft_intermediate_size

        config.latent_initializer = latent_initializer
        config.H_freq = H_freq
        config.L_freq = L_freq

    def bind_model(self, load_device='last_device'):
        target_hidden_size = self.get_hidden_size()
        draft_hidden_size = self.draft_model.get_hidden_size()

        if self.config.latent_initializer == 'learned':
            self.draft_model.fc_init = torch.nn.Linear(
                2 * target_hidden_size, draft_hidden_size, bias=True
            )

        self.draft_model.fc = torch.nn.Linear(
            2 * target_hidden_size + draft_hidden_size, draft_hidden_size, bias=True
        )

        self.draft_model.l2s = torch.nn.Linear(
            draft_hidden_size, target_hidden_size, bias=False
        )

        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

        if load_device == 'last_device':
            if len(self.get_base_layers()) > 0:
                last_device = next(self.get_base_layers()[-1].parameters()).device
            else:
                last_device = self.device
            load_device = last_device
            self.draft_model.to(device=load_device, dtype=self.base_model.dtype)

    def speculative_forward(self, input_ids, encoder_outputs, target_hiddens, **kwargs):
        bs, seq_len = encoder_outputs.shape[0], encoder_outputs.shape[1]
        with torch.no_grad():
            inputs_embeds = self.get_token_embedding(input_ids)
            device, dtype = inputs_embeds.device, inputs_embeds.dtype
            states = encoder_outputs.to(device=device, dtype=dtype)

        # (kind of) TTT loop
        for _ in range(self.config.H_freq):
            if self.config.latent_initializer == 'random':
                z = torch.rand(bs, seq_len, self.config.draft_hidden_size)
                init_latents = ((z - 0.5) * 0.2 * 512 / seq_len).to(states.device)
            elif self.config.latent_initializer == 'learned':
                init_latents = self.draft_model.fc_init(
                    torch.cat((inputs_embeds, states), dim=-1)
                )
            else:
                raise NotImplementedError

            # RNN loop
            for j in range(self.config.L_freq):
                if j == 0:
                    latents = init_latents
                else:
                    latents = torch.cat((init_latents[:, :1, :], latents[:, :-1, :]), dim=1)

                latent_inputs_embeds = self.draft_model.fc(
                    torch.cat((inputs_embeds, states, latents), dim=-1)
                )

                decoder_outputs = self.draft_model(
                    inputs_embeds=latent_inputs_embeds,
                    use_cache=False, **kwargs,
                )
                latents = decoder_outputs[0].to(target_hiddens.device)

            states = self.draft_model.l2s(latents)
        predict = states

        # Below is kept the same as EagleV2, to control variables.
        with torch.no_grad():
            target_logits = self.get_token_logits(target_hiddens)
            pred_logits = self.get_token_logits(predict)

        labels = kwargs['labels']
        loss_mask = (labels != -100)
        loss_mask[:, -1] = 0
        loss_mask = loss_mask[:, :, None]
        num_items_in_batch = kwargs.get('num_items_in_batch', loss_mask.sum())

        pred_logits = pred_logits.to(target_hiddens.device)
        target_logits = target_logits.to(target_hiddens.device)
        pred_logp = torch.nn.LogSoftmax(dim=2)(pred_logits)
        target_p = torch.nn.Softmax(dim=2)(target_logits)
        plogp = target_p * pred_logp
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (num_items_in_batch + 1e-5)

        vloss = self.smooth_l1(predict, target_hiddens)
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (num_items_in_batch + 1e-5)

        loss = self.config.ploss_w * ploss + self.config.vloss_w * vloss

        ## in-batch accuracy evaluation/print
        base_logits = self.get_token_logits(encoder_outputs)
        base_labels = base_logits.argmax(dim=-1)
        base_labels[labels == -100] = -100
        corrects, total = prediction_accuracy(
                              pred_logits[:, :-1], base_labels[:, 1:])
        topk_accuracy = corrects / (total + 1e-5)
        if random.random() < 0.02:
            print_predictions(self.tokenizer, input_ids[0, :-1],
                              pred_logits[0, :-1], base_labels[0, 1:])
        return (
            dict(loss=loss, decoder_outputs=decoder_outputs, attention_mask=kwargs['attention_mask']),
            dict(loss=loss, ploss=ploss, vloss=vloss,
                 _num_items_in_batch=num_items_in_batch, avg_topk_accuracy=topk_accuracy)
        )
