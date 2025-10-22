import os
import torch
from datasets import Dataset, load_dataset
from functools import partial
from specforge_het.data_load__eagle import data_load as data_load__eagle


def read_tensor(path, max_length):
    tensor = torch.load(path, weights_only=True)
    return tensor[:max_length, :]


def pad_tensors(tensors, padding_side='left'):
    max_len = max([t.shape[0] for t in tensors])
    new_tensors = []
    for t in tensors:
        pad_len = max_len - t.shape[0]
        if padding_side == 'left':
            t = torch.nn.functional.pad(t, (0, 0, pad_len, 0))
        else:
            t = torch.nn.functional.pad(t, (0, 0, 0, pad_len))
        new_tensors.append(t.unsqueeze(0))
    return torch.cat(new_tensors)


def collator_fn(dataset_path, tokenizer, max_length, batch):
    input_ids = [j['input_ids'][:max_length] for j in batch]
    labels = [j['labels'][:max_length] for j in batch]
    data_ids = [j['data_ids'] for j in batch]
    dry_run = False
    if 'tensor_path' in batch[0] and not dry_run:
        tensors = [read_tensor(
            os.path.join(dataset_path, 'tensors', j['tensor_path']),
            max_length
        ) for j in batch]
        tensors = pad_tensors(tensors)
    else:
        tensors = None

    inp = tokenizer.pad(dict(input_ids=input_ids), padding_side='left', return_tensors='pt')
    input_ids = inp.input_ids
    attention_mask = inp.attention_mask
    lab = tokenizer.pad(dict(input_ids=labels), padding_side='left', return_tensors='pt')
    labels = lab.input_ids
    labels[attention_mask == 0] = -100

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_outputs=tensors,
        labels=labels,
        data_ids=data_ids,
    )


def collator_raw_fn(dataset_path, tokenizer, max_length, batch):
    from .data_gen import collator_fn as data_gen_collator_fn
    return data_gen_collator_fn(tokenizer, max_length, batch, verbose=False)


def data_load(dataset_configs):
    if dataset_configs.read_eagle_format:
        return data_load__eagle(dataset_configs)

    if os.path.exists(dataset_configs.path):
        train_dataset = Dataset.load_from_disk(
            dataset_path=dataset_configs.path
        )
        train_collator = partial(collator_fn, dataset_configs.path)
    else:
        dataset_path = dict(
            path="Aeala/ShareGPT_Vicuna_unfiltered",
            data_files=["ShareGPT_V4.3_unfiltered_cleaned_split.json"],
            revision='8b0048ad6ae8c22f46a78c15559dec98feef5539'
        )
        train_dataset = load_dataset(**dataset_path)['train']

        from .data_gen import map_shargpt_fn, filter_sharegpt_fn
        train_dataset = train_dataset.map(map_shargpt_fn)
        train_dataset = train_dataset.filter(filter_sharegpt_fn)

        train_collator = partial(collator_raw_fn, dataset_configs.path)

    if dataset_configs.eval_path is not None:
        assert os.path.exists(dataset_configs.eval_path)
        eval_dataset = Dataset.load_from_disk(
            dataset_path=dataset_configs.eval_path
        )
        eval_collator = partial(collator_fn, dataset_configs.eval_path)
    else:
        eval_dataset, eval_collator = None, None

    samples = dataset_configs.manual_sample_ids
    if samples:
        train_dataset = train_dataset.filter(lambda x: x['data_ids'] in samples)
        train_dataset = train_dataset.sort("data_ids", reverse=True)

    max_read_items = dataset_configs.max_read_items
    if max_read_items:
        train_dataset = train_dataset.select(range(max_read_items))

    print('[training data preview]', train_dataset[:10]['data_ids'], '...')
    return [train_dataset, train_collator, eval_dataset, eval_collator]
