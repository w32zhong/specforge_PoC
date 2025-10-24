import os
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from typing import Any, Dict, List


def list_files(path, max_cnt):
    datapath = []
    good, bad = 0, 0
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                #data = torch.load(file_path)
                datapath.append(file_path)
                good += 1
                if max_cnt and good >= max_cnt:
                    return datapath
            except Exception as e:
                print(e, file_path)
                bad += 1
    print('good/bad:', good, bad)
    return list(sorted(datapath))


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:2048][None, :]
        input_ids = data['input_ids'][:2048][None, :]
        loss_mask = data["loss_mask"][:2048][None, :] # 1 means agent msg

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["data_ids"] = self.data[index]
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, ignore_, ignore__, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])

        batch_data_ids = [item["data_ids"] for item in features]
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)

        # [BEAGLE] not using: batch_target.
        labels = batch_input_ids.clone()
        labels[batch_loss_mask.bool() == False] = -100
        return dict(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            encoder_outputs=batch_hidden_states,
            target_hiddens=batch_target,
            labels=labels,
            loss_mask=batch_loss_mask,
            data_ids=batch_data_ids
        )


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


def data_load(dataset_configs):
    datapath = list_files(dataset_configs.path, dataset_configs.max_read_items)
    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]

    aug = AddUniformNoise(std=0.2)
    traindataset = CustomDataset(traindatapath, transform=aug)
    #traindataset = CustomDataset(traindatapath, transform=None)
    testdataset = CustomDataset(testdatapath)
    return [traindataset, DataCollatorWithPadding(), testdataset, DataCollatorWithPadding()]
