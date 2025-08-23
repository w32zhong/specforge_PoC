import torch
from transformers.cache_utils import DynamicCache


def finalize_mask(mask):
    if mask is None: return None
    inv_mask = 1.0 - mask
    final_mask = inv_mask.masked_fill(
        inv_mask.to(torch.bool), torch.finfo(mask.dtype).min)
    return final_mask


class EagleV2:
    def __init__(self, draft_layers=1, skip_input_layernorm=True, skip_output_norm=True, **kwargs):
        self.config.draft_layers = draft_layers
        self.config.skip_input_layernorm = skip_input_layernorm
        self.config.skip_output_norm = skip_output_norm

    def on_draft_model_set(self):
        if len(self.get_base_layers()) > 0:
            last_device = next(self.get_base_layers()[-1].parameters()).device
        else:
            last_device = self.device

        self.draft_model.to(last_device)

        hidden_size = self.get_hidden_size()
        self.draft_model.eagle_fc = torch.nn.Linear(
            2 * hidden_size, hidden_size, bias=True,
            device=last_device, dtype=self.base_model.dtype
        )
        self.smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

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
        with torch.no_grad():
            inputs_embeds = self.get_token_embedding(input_ids)
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        prev_states = encoder_outputs.to(device=device, dtype=dtype)
        next_states = target_hiddens.to(device=device, dtype=dtype)

        inputs_embeds_concate = self.draft_model.eagle_fc(
            torch.cat((inputs_embeds, prev_states), dim=-1)
        )

        decoder_outputs = self.draft_model(
            inputs_embeds=inputs_embeds_concate,
            **kwargs,
        )
        predict = decoder_outputs[0]

        with torch.no_grad():
            target_logits = self.get_token_logits(next_states)
            target_p = torch.nn.Softmax(dim=2)(target_logits)

        labels = kwargs['labels']
        loss_mask = (labels != -100)
        loss_mask[:, -1] = 0
        loss_mask = loss_mask[:, :, None]

        pred_logits = self.get_token_logits(predict)
        pred_logp = torch.nn.LogSoftmax(dim=2)(pred_logits)
        plogp = target_p * pred_logp
        num_items_in_batch = kwargs.get('num_items_in_batch', loss_mask.sum())
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (num_items_in_batch + 1e-5)

        ## DEBUG 
        #from specforge_het.debug import test_nan_grad
        #ploss.backward()
        #assert not test_nan_grad(self)

        vloss = self.smooth_l1(predict, next_states)
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (num_items_in_batch + 1e-5)

        #loss = ploss + 10 * vloss # align with LM losses instead of (0.1 * ploss + vloss)
        loss = 0.1 * ploss + vloss

        return (
            dict(loss=loss, decoder_outputs=decoder_outputs, attention_mask=kwargs['attention_mask']),
            dict(loss=loss, ploss=ploss, vloss=vloss, _num_items_in_batch=num_items_in_batch)
        )

    @staticmethod
    def position_embeddings_slice(position_embeddings, slice):
        return position_embeddings[0][:, slice, :], position_embeddings[1][:, slice, :]

    def sample_tokens(self, hiddens):
        logits = self.get_token_logits(hiddens)
        return logits.argmax(dim=-1)

    def speculative_generate(self, input_ids, attention_mask, **kwargs):
        inputs_embeds = self.get_token_embedding(input_ids)
        encoder_hidden_states, base_kv, draft_kv = self.prefill(inputs_embeds, attention_mask)

        tokens = self.sample_tokens(encoder_hidden_states[:, -1:, :])
        next_root = tokens[0, 0].item()
        print(self.tokenizer.batch_decode(input_ids)[0], end='', flush=True)
        print(self.tokenizer.batch_decode(tokens)[0], end='', flush=True)

        length = input_ids.shape[-1]
        max_length = min(self.get_max_ctx_length(), self.inference_configs.max_length)
        cache_position = torch.arange(0, max_length, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.get_positional_embedding(inputs_embeds, position_ids)

        draft_indices = self.build_dynamic_draft_indices()

        verified_tokens = input_ids[0].tolist()
        n_old_tokens = len(verified_tokens)
        verified_tokens.append(next_root)

        while True:
            past_seq_len = base_kv.get_seq_length()
            res = self.dynamic_draft(
                next_root, draft_kv, position_embeddings,
                past_seq_len, encoder_hidden_states, **draft_indices
            )
            #tree_Q, leaf_root_paths, tree_attention, tree_positions = res
            #expand_tree_attn = F.pad(
            #    tree_attention.to(device=self.device, dtype=self.dtype),
            #    (past_seq_len, 0), value=1
            #)
            #attention_mask = finalize_mask(expand_tree_attn)[None, None]
            #q_cos, q_sin = (
            #    position_embeddings[0][:, past_seq_len + tree_positions, :],
            #    position_embeddings[1][:, past_seq_len + tree_positions, :]
            #)


    def prefill(self, inputs_embeds, attention_mask):
        base_kv = DynamicCache()
        base_outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=base_kv,
            output_hidden_states=True,
            return_dict=True
        )
        encoder_hidden_states = base_outputs.last_hidden_state
        prev_states = encoder_hidden_states[:, :-1, :].to(self.draft_model.device)
        inputs_embeds = inputs_embeds[:, 1:, :].to(self.draft_model.device)
        attention_mask = attention_mask[:, 1:]

        inputs_embeds_concate = self.draft_model.eagle_fc(
            torch.cat((inputs_embeds, prev_states), dim=-1)
        )

        draft_kv = DynamicCache()
        _ = self.draft_model(
            inputs_embeds=inputs_embeds_concate,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=draft_kv
        )

        return encoder_hidden_states, base_kv, draft_kv

    def build_dynamic_draft_indices(self):
        top_k = self.inference_configs.dynamic_draft_top_k
        all_top_k = self.inference_configs.dynamic_draft_all_top_k
        max_depth = self.inference_configs.dynamic_draft_max_depth
        device = self.draft_model.device

        zero_posi = torch.zeros(top_k, device=device, dtype=torch.long)
        logsoftmax = torch.nn.LogSoftmax(dim=-1)
        eye = torch.eye(top_k, device=device, dtype=self.dtype)
        attn_eye = finalize_mask(eye)
        path_top_idx = torch.zeros(top_k, device=device, dtype=torch.long)
        tree_size = all_top_k + 1
        tree_token_Q = torch.zeros((tree_size + 1,), dtype=torch.long, device=device)
        tree_token_Q[-1] = self.tokenizer.unk_token_id
        return dict(top_k=top_k, all_top_k=all_top_k, max_depth=max_depth,
                    zero_posi=zero_posi, logsoftmax=logsoftmax, attn_eye=attn_eye,
                    path_top_idx=path_top_idx, tree_token_Q=tree_token_Q)

    def dynamic_draft(self, next_root, draft_kv,
        position_embeddings, past_seq_len, encoder_hidden_states, *,
        top_k=None, all_top_k=None, max_depth=None, zero_posi=None,
        logsoftmax=None, attn_eye=None, path_top_idx=None, tree_token_Q=None):

        device = self.draft_model.device
        tree_size = all_top_k + 1
        cos, sin = position_embeddings

        logprobs_Q = [torch.zeros((1,), device=device)]
        tree_mask = torch.zeros((1, past_seq_len + 1),
            device=device, dtype=encoder_hidden_states.dtype)
        attention_mask = tree_mask[None, None]

        draft_Q, parent_Q, score_Q = [], [], []
        draft_tokens = torch.tensor([[next_root]], device=device)

        for depth in range(max_depth):
            inputs_embeds = self.get_token_embedding(draft_tokens)
            n_draft_tokens = inputs_embeds.shape[1]

            index = (past_seq_len + depth) + zero_posi[:n_draft_tokens]
            position_embeddings = (
                cos[:, index, :],
                sin[:, index, :]
            )

            hidden_states = inputs_embeds.to(device)
            for decoder_layer in self.draft_model.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                    past_key_value=draft_kv,
                )
            hidden_states = self.draft_model.norm(hidden_states)

            logits = self.get_token_logits(hidden_states)
            logprobs = logsoftmax(logits).to(device)
            top = torch.topk(logprobs, top_k, dim=-1)
            top_tokens, top_logprobs = top.indices, top.values

            # (1, )  <= initial state
            # (10,)  <= topk ==  (1, 10)    +  (1, 1)
            # (10,)  <= topk ==  (10, 10)   +  (10, 1)
            path_logprobs = top_logprobs[0] + logprobs_Q[-1][:, None]

            draft_Q.append(top_tokens[0])
            score_Q.append(path_logprobs)
            if depth == 0:
                parent_Q.append(path_top_idx - 1)
            elif depth == 1:
                parent_Q.append(path_top_idx)
            else:
                parent_Q.append(top_k + path_top_idx + (depth - 2)*(top_k**2))

            path_top = torch.topk(path_logprobs.view(-1), top_k, dim=-1)
            path_top_idx, path_top_logprobs = path_top.indices, path_top.values

            draft_tokens = top_tokens.view(-1)[path_top_idx][None]
            logprobs_Q.append(path_top_logprobs)

            parent_idx = path_top_idx // top_k
            tree_mask = torch.cat((tree_mask[parent_idx], attn_eye), dim=-1)

        breakpoint()

        cat_parent = torch.cat(parent_Q, dim=0)[top_k-1:].tolist() # [1 + (depth-1) x10]
        cat_tokens = torch.cat(draft_Q, dim=0).view(-1) # [10 + (depth-1) x100]
        cat_scores = torch.cat(score_Q, dim=0).view(-1) # [10 + (depth-1) x100]
        # p self.tokenizer.decode([cat_tokens[2]])

        all_top = torch.topk(cat_scores, all_top_k, dim=-1)

        tree_token_Q[0] = next_root
        tree_token_Q[1: tree_size] = cat_tokens[all_top.indices]

        def node_str(node):
            if node == -1:
                return '<root>'
            else:
                score = cat_scores[node]
                token = self.tokenizer.decode([cat_tokens[node]])
                return f'#{node} {printable(token)}({score:.1f})'

        tree_node_Q = [-1] + all_top.indices.tolist() # guess nodes
        node2idx = {node: idx for idx, node in enumerate(tree_node_Q)}
        tree_positions = torch.zeros(tree_size, dtype=torch.long)
        attention_mask = torch.eye(tree_size, dtype=torch.long)
        attention_mask[:, 0] = 1
        visited = torch.zeros(tree_size, dtype=torch.long)
        visited[0] = 1
        for row in range(1, tree_size):
            node, depth = tree_node_Q[row], 1
            visited[node2idx[node]] += 1
            if debug: print('row:', node_str(node), end=' ')
            while True:
                node = cat_parent[node // top_k]
                depth += 1
                if debug: print('->', node_str(node), end=' ')
                if node == -1:
                    visited[0] += 1
                    break
                idx = node2idx[node]
                visited[idx] += 1
                attention_mask[row, idx] = 1
            tree_positions[row] = depth - 1
            if debug: print()
        leaves = ~(visited - 1).bool()
        leaf_depths = (tree_positions[leaves] + 1).tolist()

        if debug:
            torch.set_printoptions(profile="full")
            torch.set_printoptions(linewidth=800)
            print(attention_mask, attention_mask.shape)
            print(tree_positions)

        leaf_root_paths = torch.zeros(
            (leaves.sum().item(), max(leaf_depths) + 1), dtype=torch.long
        ) - 1

        tree_node_Q = torch.tensor(tree_node_Q)
        leaf_nodes = tree_node_Q[leaves].tolist() # to get items in non-tensor
        for i, (node, depth) in enumerate(zip(leaf_nodes, leaf_depths)):
            cnt = 1
            while True:
                idx = node2idx[node]
                leaf_root_paths[i, depth - cnt] = idx
                cnt += 1
                node = cat_parent[node // top_k]
                if node == -1:
                    leaf_root_paths[i, 0] = 0
                    break
        leaf_root_paths = leaf_root_paths.to(device=tree_token_Q.device)

        return tree_token_Q, leaf_root_paths, attention_mask, tree_positions
