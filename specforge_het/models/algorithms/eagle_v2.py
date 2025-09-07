import torch
from transformers.cache_utils import DynamicCache


def finalize_mask(mask): # 0 -> -infty,  1 -> 0
    if mask is None: return None
    inv_mask = 1.0 - mask
    final_mask = inv_mask.masked_fill(
        inv_mask.to(torch.bool), torch.finfo(mask.dtype).min)
    return final_mask


def shrink_cache(cache, past_seq_len, select_indices=None, debug=False):
    if debug: print(cache[0, 0].sum(-1))
    if select_indices is not None:
        B, S = select_indices.shape
        B, H, _, D = cache.shape
        select_indices = select_indices.to(cache.device)

        idx = select_indices[:, None, :, None].expand(B, H, S, D)
        src = torch.gather(cache, dim=2, index=idx)

        dst = cache[..., past_seq_len: past_seq_len + S, :]
        dst.copy_(src, non_blocking=False)

        new_cache = cache[..., : past_seq_len + S, :]
    else:
        new_cache = cache[..., : past_seq_len, :]
    if debug: print(new_cache[0, 0].sum(-1))
    return new_cache


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
        assert encoder_outputs.shape[-1] == self.get_hidden_size(), (
            "input and pre-generated hidden space mismatch! Wrong training dataset?"
        )

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

    ###############
    ## Inference ##
    ###############

    def prefill_base_model(self, inputs_embeds, attention_mask):
        base_kv = DynamicCache()
        base_outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=base_kv,
            output_hidden_states=True,
            return_dict=True
        )
        assert base_kv.get_seq_length() > 0
        return base_outputs.last_hidden_state, base_kv

    def prefill_draft_model(self, inputs_embeds, prev_states, *,
                            attention_mask=None, draft_kv=None):
        prev_states = prev_states.to(self.draft_model.device)
        inputs_embeds = inputs_embeds.to(self.draft_model.device)

        inputs_embeds_concate = self.draft_model.eagle_fc(
            torch.cat((inputs_embeds, prev_states), dim=-1)
        )

        draft_kv = DynamicCache() if draft_kv is None else draft_kv
        _ = self.draft_model(
            inputs_embeds=inputs_embeds_concate,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=draft_kv
        )
        return draft_kv

    def prebuilt_position_embeddings(self, inputs_embeds):
        max_length = min(self.get_max_ctx_length(), self.inference_configs.max_length)
        cache_position = torch.arange(0, max_length, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        return self.get_positional_embedding(inputs_embeds, position_ids)

    def speculative_generate(self, input_ids, attention_mask, **kwargs):
        inputs_embeds = self.get_token_embedding(input_ids)
        position_embeddings = self.prebuilt_position_embeddings(inputs_embeds)
        max_length = position_embeddings[0].shape[1]

        hidden_states, base_kv = self.prefill_base_model(inputs_embeds, attention_mask)
        prev_states, last_states = hidden_states[:, :-1], hidden_states[:, -1:]

        logits = self.get_token_logits(last_states)
        next_root = logits.argmax(dim=-1)
        yield next_root

        draft_kv = self.prefill_draft_model(inputs_embeds[:, 1:], prev_states,
                                            attention_mask=attention_mask[:, 1:])

        #  pre-fill  |draft tree | verify |next_root
        # h1 h2 h3 h4|~~~~~~~~~~ |h5 h6 h7|
        #   \draft\  |           |        |
        # t1 t2 t3 t4|t5 ~~~~~~~~|t5 t6 t7|t8
        draft_indices = self.dynamic_draft_indices(input_ids.shape[0])
        while True:
            # past_seq_len is derived from base_kv because
            # base_kv length is more stable.
            past_seq_len = base_kv.get_seq_length()

            draft_outputs = self.dynamic_draft(
                draft_kv, position_embeddings, past_seq_len - 1,
                next_root, last_states, **draft_indices
            )
            accept_tokens, accept_idx, hidden_states = self.verify(
                base_kv, position_embeddings, past_seq_len,
                **draft_outputs
            )

            #print(draft_kv.layers[0].keys[0,0].sum(-1))
            self.reset_kv_cache(base_kv, draft_kv, past_seq_len, accept_idx)
            #print(draft_kv.layers[0].keys[0,0].sum(-1))

            yield accept_tokens

            # TODO: last states per batch is not necessarily the last element.
            if accept_tokens.shape[-1] > 1:
                prev_states, last_states = hidden_states[:, :-1], hidden_states[:, -1:]
                prefill_tokens, next_root = accept_tokens[:, :-1], accept_tokens[:, -1:]
                inputs_embeds = self.get_token_embedding(prefill_tokens)
                draft_kv = self.prefill_draft_model(inputs_embeds, prev_states,
                                                    draft_kv=draft_kv)
            else:
                last_states = hidden_states
                next_root = accept_tokens

    def topk_tokens(self, hiddens, top_k, debug=False):
        logits = self.get_token_logits(hiddens)
        logsoftmax = torch.nn.LogSoftmax(dim=-1)
        logprobs = logsoftmax(logits).to(self.draft_model.device)
        top = torch.topk(logprobs, top_k, dim=-1)
        top_tokens, top_logprobs = top.indices, top.values
        if debug:
            for i in range(top_tokens.shape[1]):
                print([self.tokenizer.decode(t) for t in top_tokens[0, i]])
            print('-' * 30)
        return top_tokens, top_logprobs

    # for top_k = 2
    #          top_tokens  top_path_idx  parent    cat_parent
    #            R (root)                          [-1,
    # depth 0: 0   1       [-1, -1]      [-1, -1]   0, 1,
    # depth 1: X X 4 5     [0, 1]        [0, 1]     5, 4,
    # depth 2: X 7 8 X     [3, 2]  +2  = [5, 4]     8, 7]
    # depth 3:  ...        [2, 1]  +2+4= [8, 7]
    #         (^-- Big X denotes unselected for expansion)
    #
    # We can conclude: parent_ID = cat_parent[node_ID // top_k])
    # Because:
    #  1. For cat_parent[i], the corresponding node n is
    #    * (i - 1) * 2 - 2 + 0  (the left one)
    #    * (i - 1) * 2 - 2 + 1  (the right one)
    #  2. The children node of n, in each case, are
    #    * n + 2^2 + 0 and n + 2^2 + 1
    #    * n + 2^2 + 0 and n + 2^2 - 1
    #  3. The corresponding child node c of i is then
    #    * (i - 1) * 2 + 2 + 0/1  = 2 * i + 0/1
    #    * (i - 1) * 2 + 3 + 0/-1 = 2 * i + 1/0
    #  4. Dividing by 2, we get
    #     c // 2 = i
    def get_all_top_nodes(self, parent_Q, draft_Q, score_Q, top_k, all_top_k):
        B = draft_Q[0].shape[0]
        cat_parent = torch.cat(parent_Q, dim=1)[:, top_k - 1:] # [B, 1 + (max_depth - 1) x top_k]
        cat_tokens = torch.cat(draft_Q, dim=1).view(B, -1) # [B, top_k + (max_depth - 1) x top_k^2]
        cat_scores = torch.cat(score_Q, dim=1).view(B, -1) # [B, top_k + (max_depth - 1) x top_k^2]

        top_nodes = torch.topk(cat_scores, all_top_k, dim=-1).indices # [B, all_top_k]
        ranked_top_nodes = torch.sort(top_nodes, dim=-1).values # for later getting masks bottom-up
        return ranked_top_nodes, cat_parent, cat_tokens

    @staticmethod
    def construct_row_map(ranked_top_nodes, ranked_top_nodes_parents):
        B = ranked_top_nodes.shape[0]
        row_map = torch.searchsorted(ranked_top_nodes, ranked_top_nodes_parents)
        row_map[ranked_top_nodes_parents == -1] = -1
        # move the for-loop below to CPU to avoid multiple memory movements
        row_map = row_map.cpu()
        # to include draft root: prepend zeros and offset-by-one:
        zeros = torch.zeros((B, 1), dtype=torch.long)
        row_map = torch.cat((zeros, row_map + 1), dim=-1)
        return row_map

    # for top_k = 2
    #          top_tokens  top_path_idx  parent    cat_parent
    #            R (root)                          [-1,
    # depth 0: 0   1       [-1, -1]      [-1, -1]   0, 1,
    # depth 1: X X 4 5     [0, 1]        [0, 1]     5, 4,
    # depth 2: X 7 8 X     [3, 2]  +2  = [5, 4]     8, 7]
    # depth 3:  ...        [2, 1]  +2+4= [8, 7]
    #         (^-- Big X denotes unselected for expansion)
    #
    # We want to extract the tree mask associated to selected nodes:
    #       R 0 1 4 5 7 8
    #       -------------
    #    R |1
    #    0 |1 1
    #    1 |1   1
    #    4 |1   1 1
    #    5 |1   1   1
    #    7 |1     1   1
    #    8 |1       1   1
    @staticmethod
    def extract_tree_mask(idx_B, row_map, all_top_k):
        B = row_map.shape[0]
        tree_mask = torch.eye(all_top_k + 1).int()
        tree_mask[:, 0] = 1
        tree_mask = tree_mask.expand(B, -1, -1)
        for row in range(1, all_top_k + 1):
            parent_row = tree_mask[idx_B, row_map[:, row]]
            tree_mask[idx_B, row] |= parent_row

        tree_positions = torch.sum(tree_mask, dim=-1) - 1
        return tree_mask, tree_positions

    @staticmethod
    def extract_leaf_root_paths(row_map, max_depth, tree_mask, tree_positions):
        col_sum = tree_mask.sum(dim=1)
        is_leaf = (col_sum == 1)
        n_leaves, leaves = is_leaf.sum(dim=-1), is_leaf.nonzero()
        # leaves: (batch, row) pairs

        B = row_map.shape[0]
        leaf_root_paths = torch.zeros(
            B, n_leaves.max().item(), max_depth + 1, dtype=torch.long
        ) - 1
        L = leaves.size(0)
        idx_L = torch.arange(L)

        curr_batches, curr_rows = leaves[:, 0], leaves[:, 1]
        for _ in range(max_depth + 1):
            depths = tree_positions[curr_batches, curr_rows]
            # leaf_root_paths.shape: (B, L, max_depth + 1)
            # curr_batches, curr_rows, idx_L, depths shape: (L)
            leaf_root_paths[curr_batches, idx_L, depths] = curr_rows
            curr_rows = row_map[curr_batches, curr_rows]
        return leaf_root_paths

    def dynamic_draft_indices(self, B):
        top_k = self.inference_configs.dynamic_draft_top_k
        all_top_k = self.inference_configs.dynamic_draft_all_top_k
        max_depth = self.inference_configs.dynamic_draft_max_depth
        device = self.draft_model.device

        zero_posi = torch.zeros(top_k, device=device, dtype=torch.long)
        eye = torch.eye(top_k, device=device, dtype=self.dtype)
        batched_eye = eye.expand(B, -1, -1)
        attn_eye = finalize_mask(batched_eye)
        idx_B = torch.arange(B)[:, None]
        top_path_idx = torch.zeros((B, top_k), device=device, dtype=torch.long)
        tree_size = all_top_k + 1
        draft_token_buff = torch.zeros((B, tree_size + 1), dtype=torch.long,
                                       device=self.base_model.device) - 100
        return dict(top_k=top_k, all_top_k=all_top_k, max_depth=max_depth,
                    zero_posi=zero_posi, attn_eye=attn_eye, idx_B=idx_B,
                    top_path_idx=top_path_idx, draft_token_buff=draft_token_buff)

    def dynamic_draft(self, draft_kv, position_embeddings, past_seq_len,
        next_root, last_states, *, top_k, all_top_k, max_depth, zero_posi,
        attn_eye, idx_B, top_path_idx, draft_token_buff):

        B = next_root.shape[0]
        draft_device = self.draft_model.device
        tree_size = all_top_k + 1

        top_path_logprobs = torch.zeros((B, 1), device=draft_device)

        # previous context tokens are fully visible to root
        tree_depth_mask = torch.zeros((B, 1, past_seq_len + 1),
            device=draft_device, dtype=self.model.dtype)

        draft_Q, parent_Q, score_Q = [], [], []
        draft_tokens = next_root

        for depth in range(max_depth):
            # get hidden states from draft_tokens
            inputs_embeds = self.get_token_embedding(draft_tokens)
            inputs_embeds = inputs_embeds.to(draft_device) # [B, n_draft_tokens, H]
            hidden_states = self.draft_model.eagle_fc(
                torch.cat((inputs_embeds, last_states), dim=-1)
            ) # [B, n_draft_tokens, H]

            n_draft_tokens = inputs_embeds.shape[1]
            tree_depth_pos = (past_seq_len + depth) + zero_posi[:n_draft_tokens]

            for decoder_layer in self.draft_model.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=tree_depth_mask.unsqueeze(1),
                    position_embeddings=(
                        position_embeddings[0][:, tree_depth_pos, :],
                        position_embeddings[1][:, tree_depth_pos, :]
                    ),
                    use_cache=True,
                    past_key_value=draft_kv,
                )
            hidden_states = self.draft_model.norm(hidden_states)

            # get top_k tokens from *each* children of this level, use top_tokens as nodes of this level.
            top_tokens, top_logprobs = self.topk_tokens(hidden_states, top_k, debug=False)

            # top_path_logprobs initial shape: (B, 1)
            # depth = 0: (B, 1, 10) + (B, 1, 1)   --[top path]-> (B, 10)
            # depth > 0: (B, 10, 10) + (B, 10, 1) --[top path]-> (B, 10)
            top_path_logprobs = top_logprobs + top_path_logprobs.unsqueeze(-1)

            draft_Q.append(top_tokens) # token nodes of this depth
            score_Q.append(top_path_logprobs) # score nodes of this depth

            # for top_k = 2, each node can select top-2 children at each depth:
            #          top_tokens  top_path_idx  parent
            #            R (root)
            # depth 0: 0   1       [-1, -1]      [-1, -1]
            # depth 1: X X 4 5     [0, 1]        [0, 1]
            # depth 2: X 7 8 X     [3, 2]  +2  = [5, 4]
            # depth 3:  ...        [2, 1]  +2+4= [8, 7]
            #         (^-- Big X denotes unselected for expansion)
            # ...
            # for depth d:
            # [top_k + (d - 2) * beam_width] will count all nodes above d - 1,
            # and + top_path_idx will be the parents of this depth.
            # Note that this top_path_idx is obtained from the last iteration.
            if depth == 0:
                base = -1 # [-1, -1, ..., -1]
            elif depth == 1:
                base = 0 # +[0, 1, 2, ..., 9]
            else:
                beam_width = top_k ** 2
                base = top_k + (depth - 2) * beam_width
            parent_nodes = base + top_path_idx
            parent_Q.append(parent_nodes)

            # select top_k tokens out of all the nodes of this level.
            top_path = torch.topk(top_path_logprobs.view(B, -1), top_k, dim=-1)
            top_path_idx, top_path_logprobs = top_path.indices, top_path.values
            parent_idx = top_path_idx // top_k # [B, 10]

            # only the selected top_k nodes are going to be "expanded".
            draft_tokens = torch.gather(
                top_tokens.view(B, -1), dim=-1, index=top_path_idx
            ) # [B, 10]

            last_states = hidden_states[idx_B, parent_idx] # [B, 10, H]

            select_rows = tree_depth_mask[idx_B, parent_idx] # [B, 10, n_col]
            tree_depth_mask = torch.cat((select_rows, attn_eye), dim=-1)

        ranked_top_nodes, parents, tokens = self.get_all_top_nodes(
            parent_Q, draft_Q, score_Q, top_k, all_top_k
        )

        ranked_top_nodes_parents = parents[idx_B, ranked_top_nodes // top_k]
        if True:
            # Sanity Check
            # (parent scores must be strictly larger, so parents must
            # appear in all_top_k candidates if their children appear)
            parent_appears = torch.isin(ranked_top_nodes_parents, ranked_top_nodes)
            parent_is_root = (ranked_top_nodes_parents == -1)
            assert torch.all(parent_appears | parent_is_root)
        row_map = self.construct_row_map(ranked_top_nodes, ranked_top_nodes_parents)
        draft_token_buff[:, 0] = next_root
        draft_token_buff[:, 1:-1] = tokens[idx_B, ranked_top_nodes]

        tree_mask, tree_positions = self.extract_tree_mask(idx_B, row_map, all_top_k)
        #torch.set_printoptions(threshold=torch.inf)
        #torch.set_printoptions(linewidth=200)
        #print(tree_mask)
        assert tree_positions.max() <= max_depth, (
            f'tree_positions={tree_positions.max()} > max_depth={max_depth}'
        )

        leaf_root_paths = self.extract_leaf_root_paths(row_map, max_depth,
                                                      tree_mask, tree_positions)

        return dict(
            draft_token_buff=draft_token_buff, tree_mask=tree_mask,
            tree_positions=tree_positions, leaf_root_paths=leaf_root_paths
        )

    def verify(self, base_kv, position_embeddings, past_seq_len, *,
               draft_token_buff, tree_mask, tree_positions, leaf_root_paths):
        B = draft_token_buff.shape[0]
        idx_B = torch.arange(B).unsqueeze(-1)

        ext_tree_mask = torch.nn.functional.pad(
            tree_mask, (past_seq_len, 0), value=1
        ).float()

        ext_tree_positions = past_seq_len + tree_positions

        hidden_states = self.get_token_embedding(draft_token_buff[..., :-1])
        assert len(self.base_model.layers) == self.config.num_hidden_layers
        for decoder_layer in self.base_model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=finalize_mask(ext_tree_mask).unsqueeze(1),
                position_embeddings=(
                    position_embeddings[0][idx_B, ext_tree_positions],
                    position_embeddings[1][idx_B, ext_tree_positions]
                ),
                use_cache=True,
                past_key_value=base_kv
            )
        hidden_states = self.base_model.norm(hidden_states)

        logits = self.get_token_logits(hidden_states)
        next_tokens = logits.argmax(dim=-1)

        next_tokens = next_tokens.to(self.base_model.device)
        leaf_root_paths = leaf_root_paths.to(self.base_model.device)

        draft_paths = draft_token_buff[idx_B, leaf_root_paths][..., 1:]
        truth_paths = next_tokens[idx_B, leaf_root_paths]

        match_paths = (draft_paths == truth_paths[..., :-1]).cumprod(dim=-1)
        match_length, match_idx = match_paths.sum(-1).topk(k=1)
        # since we only care about top-1 here, no need for the last dimension.
        idx_B, match_length, match_idx, bonus = (
            idx_B.squeeze(-1), match_length.squeeze(-1), match_idx.squeeze(-1), 1
        ) # shape: [B]
        accept_tokens = truth_paths[idx_B, match_idx, :match_length + bonus]
        accept_idx = leaf_root_paths[idx_B, match_idx, :match_length + bonus]
        accept_states = hidden_states[idx_B, accept_idx.to(hidden_states.device)]
        return accept_tokens, accept_idx, accept_states

    def reset_kv_cache(self, base_kv, draft_kv, past_seq_len, indices):
        select_indices = past_seq_len + indices
        for layer in base_kv.layers:
            layer.keys = shrink_cache(layer.keys, past_seq_len, select_indices)
            layer.values = shrink_cache(layer.values, past_seq_len, select_indices)
        for layer in draft_kv.layers:
            layer.keys = shrink_cache(layer.keys, past_seq_len)
            layer.values = shrink_cache(layer.values, past_seq_len)
