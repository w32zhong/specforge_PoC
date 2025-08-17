import os
import sys
import torch
import fcntl
from tqdm import tqdm
import random
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, Back, Style
from functools import partial

from specforge_het.configs import Configs
from specforge_het.model_load import load_models


def map_shargpt_fn(j):
    conversations = j['conversations']

    if conversations[0]['from'] == 'gpt':
        conversations = conversations[1:]

    def formatter(conv):
        if conv['from'] == 'gpt':
            role = 'assistant'
        elif conv['from'] == 'human':
            role = 'user'
        else:
            assert ValueError
        content = conv['value']
        assert isinstance(content, str)
        return dict(role=role, content=content)
    new_conversations = list(map(formatter, conversations))
    return dict(conversations=new_conversations)


def filter_sharegpt_fn(j):
    conversations = j['conversations']
    if len(conversations) < 2:
        print('skipping:', j['id'])
        return False
    return True


def debug_print_chat_turns(tokenizer, input_ids, ans_mask):
    for tok_id, is_ans in zip(input_ids, ans_mask):
        if is_ans.bool().item():
            print(Fore.BLUE, end='')
        else:
            print(Fore.WHITE + Style.DIM, end='')
        print(tokenizer.decode([tok_id.item()]), end=' ' + Style.RESET_ALL)
    print()


def collator_fn(tokenizer, max_length, batch, verbose=True):
    conversations = [j['conversations'] for j in batch]
    data_ids = [j['id'] for j in batch]
    inp_list, ans_mask = [], []
    for id, conv in zip(data_ids, conversations):
        conv_chat = tokenizer.apply_chat_template(conv,
            tokenize=True, max_length=max_length, truncation=True,
            return_assistant_tokens_mask=True, return_dict=True
        )
        inp_list.append(conv_chat.input_ids)
        ans_mask.append(conv_chat.assistant_masks)

    inp = tokenizer.pad(dict(input_ids=inp_list), padding_side='left', return_tensors='pt')
    ans = tokenizer.pad(dict(input_ids=ans_mask), padding_side='left', return_tensors='pt')
    input_ids = inp.input_ids
    attention_mask = inp.attention_mask
    answer_mask = ans.input_ids
    answer_mask[answer_mask == tokenizer.pad_token_id] = 0
    labels = input_ids.clone()
    labels[0 == answer_mask] = -100
    if torch.all(answer_mask == 0).item():
        return None
    if verbose and random.random() < 0.05:
        debug_print_chat_turns(tokenizer, input_ids[0], answer_mask[0])
        if verbose is not True: print(verbose)
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        data_ids=data_ids
    )


def save_dataset_from_array(array, path):
    def gen():
        for d in array:
            yield d;
    with open('dataset.lock', "w") as fh:
        print('obtaining lock ...')
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        print(Fore.RED, 'saving dataset ...', Style.RESET_ALL)
        ds = Dataset.from_generator(gen)
        ds.save_to_disk(path)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def save_tensor(parent_dir, name, tensor):
    name_id = hash(name) % 100
    full_dir = os.path.join(parent_dir, f'{name_id}')
    os.makedirs(full_dir, exist_ok=True)
    full_path = f'{full_dir}/{name}.pt'
    torch.save(tensor, full_path)
    rel_path = f'{name_id}/{name}.pt'
    return rel_path


def gen_dataset(config_file='configs.ini', ds_range=(sys.maxsize,), **injects):
    configs = Configs.from_config_file(config_file, **injects)
    tokenizer_path = configs.modeling.tokenizer_path
    model_path = configs.modeling.model_path
    dataset_path = configs.dataset.path

    config = configs.dataset_generation
    save_every = config.save_every
    batch_size = config.batch_size
    max_length = config.max_length
    ds_prefix = config.ds_prefix
    target = config.debug_target
    seed = config.seed
    output_dir = config.output_dir
    sharegpt_path = config.sharegpt_path

    ds_range = range(*ds_range)
    name = ds_prefix + os.path.basename(tokenizer_path)
    path = f"{output_dir}/datasets/{name}"
    ds_path = path + f'__{ds_range}'
    memo_data_rows = []
    existing_sample_ids = set()
    try:
        with open('dataset.lock', "w") as fh:
            print('obtaining lock ...')
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            print(Fore.RED, 'reading dataset ...', Style.RESET_ALL)
            ds = Dataset.load_from_disk(ds_path)
            collated_data = ds.to_dict()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

        for i in range(len(collated_data['data_ids'])):
            j = {k: collated_data[k][i] for k in collated_data.keys()}
            memo_data_rows.append(j)
        existing_sample_ids = set(collated_data['data_ids'])
        print(Fore.RED, f'existing dataset: {len(memo_data_rows)}', Style.RESET_ALL)
    except FileNotFoundError as e:
        print('Exception', e)
        pass

    tokenizer, model = load_models(configs.modeling)
    model = model.base_model

    if sharegpt_path == 'Aeala/ShareGPT_Vicuna_unfiltered':
        dataset_path = dict(
            path="Aeala/ShareGPT_Vicuna_unfiltered",
            data_files=["ShareGPT_V4.3_unfiltered_cleaned_split.json"],
            revision='8b0048ad6ae8c22f46a78c15559dec98feef5539'
        )
        dataset = load_dataset(**dataset_path)['train']
    else:
        try:
            dataset = Dataset.load_from_disk(sharegpt_path)
        except FileNotFoundError:
            dataset = load_dataset(path=sharegpt_path)['train']

    dataset = dataset.map(map_shargpt_fn)
    dataset = dataset.filter(filter_sharegpt_fn)
    dataset = dataset.shuffle(seed=seed)

    iterator = dataset.iter(batch_size=batch_size)
    progress = tqdm(total=len(dataset))
    for cnt, batched_dict in enumerate(iterator):
        batch_ids = set(batched_dict['id'])
        bs = len(batch_ids)
        progress.update(bs)
        if progress.n not in ds_range:
            continue
        if (target and target not in batch_ids) or len(batch_ids - existing_sample_ids) == 0:
            continue
        batch = [
            dict(id=id, conversations=conv)
            for id, conv in zip(batched_dict['id'], batched_dict['conversations'])
        ]
        verbose_string = f'{cnt}%{save_every}, {len(memo_data_rows)} {ds_path}'
        collated_data = collator_fn(tokenizer, max_length, batch, verbose=verbose_string)
        if collated_data is None:
            print('[skip long prompt]', [b['id'] for b in batch])
            continue

        input_ids = collated_data['input_ids']
        attention_mask = collated_data['attention_mask']
        labels = collated_data['labels']

        if target in batch_ids:
            print(batch_ids)

        with torch.no_grad():
            encoder_outputs = model(
                input_ids=input_ids.to(next(model.parameters()).device),
                attention_mask=attention_mask.to(next(model.parameters()).device),
                use_cache=False, return_dict=True
            )

        if target in batch_ids:
            breakpoint()

        last_hidden_state = encoder_outputs.last_hidden_state.cpu()
        select_fn = lambda x, m: [x[i][m[i].bool()] for i in range(bs)]
        sel_input_ids = select_fn(input_ids, attention_mask)
        sel_labels = select_fn(labels, attention_mask)
        sel_hidden_state = select_fn(last_hidden_state, attention_mask)
        for i in range(bs):
            j = {
                "input_ids": sel_input_ids[i],
                "labels": sel_labels[i],
                "data_ids": collated_data['data_ids'][i]
            }
            if j['data_ids'] in existing_sample_ids:
                continue
            j['tensor_path'] = save_tensor(os.path.join(path, 'tensors'),
                j['data_ids'], sel_hidden_state[i]
            )
            memo_data_rows.append(j)

        if (cnt + 1) % save_every == 0 and memo_data_rows:
            save_dataset_from_array(memo_data_rows, ds_path)

    if memo_data_rows:
        save_dataset_from_array(memo_data_rows, ds_path)


def push_dataset(path, name):
    dataset = DatasetDict({'train': Dataset.load_from_disk(path)})
    dataset.push_to_hub(name)


def merge_datasets(dest_path, *src_paths):
    ds_list = []
    for src_path in src_paths:
        src_ds = DatasetDict({'train': Dataset.load_from_disk(src_path)})['train']
        print(src_path, src_ds)
        ds_list.append(src_ds)
    merged_ds = concatenate_datasets(ds_list)
    merged_ds.save_to_disk(dest_path)


def probe_datasets(*paths, skip_ranged=False):
    for path in paths:
        if skip_ranged and '__range(' in path:
            continue
        try:
            ds = DatasetDict({'train': Dataset.load_from_disk(path)})['train']
            print(path, ds)
        except FileNotFoundError:
            print(path, FileNotFoundError)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(dict(
        gen_dataset=gen_dataset,
        probe_datasets=probe_datasets,
        merge_datasets=merge_datasets,
        push_dataset=push_dataset
    ))
