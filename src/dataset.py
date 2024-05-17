import json
import random
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from model import DissonanceClassifier

class DissonanceDataset(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = " ".join(dus)

                item = {
                    "id": sample['message_id'],
                    'list_pairs':[],
                    "tweet": full_message,
                    "topic": "topic",
                    "labels":[]
                }

                for pairs in sample['dissonance_pairs']:

                    du1_sent = dus[pairs['du1']]
                    du2_sent = dus[pairs['du2']]
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['list_pairs'].append([du1_sent,du2_sent])
                    item['labels'].append(disso_label)

                dataset.append(item)

            return DissonanceDataset(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "id": _inst["id"],
            "topic": _inst["topic"],
            "list_pairs": _inst["list_pairs"],
            "tweet": _inst["tweet"],
            "labels": _inst['labels']
        }

        return item


#new Dataset  with before dissconance and after Dissconance Data

class DissonanceDatasetwithbeforeafter(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = " ".join(dus)

                item = {
                    "id": sample['message_id'],
                    'list_pairs':[],
                    'before_du1':[],
                    'after_du2':[],
                    "tweet": full_message,
                    "topic": "topic",
                    "labels":[]
                }

                for pairs in sample['dissonance_pairs']:

                    du1_sent = dus[pairs['du1']]
                    du2_sent = dus[pairs['du2']]
                    before_du1= " ".join(dus[:pairs['du1']])
                    after_du2 = " ".join(dus[pairs['du2']:])
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['list_pairs'].append([du1_sent,du2_sent])
                    #item['list_pairs_before_after'].append([before_du1,after_du2])
                    item['before_du1'].append(before_du1)
                    item['after_du2'].append(after_du2)
                    item['labels'].append(disso_label)

                dataset.append(item)

            return DissonanceDatasetwithbeforeafter(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "id": _inst["id"],
            "topic": _inst["topic"],
            "list_pairs": _inst["list_pairs"],
            "tweet": _inst["tweet"],
            "labels": _inst['labels'],
            "before_du1": _inst["before_du1"],
            "after_du2": _inst['after_du2']
        }

        return item

class DissonanceDatasetwithbeforeaftersep(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = " ".join(dus)

                item = {
                    "id": sample['message_id'],
                    'list_pairs':[],
                    'before_after_sep':[],
                    "tweet": full_message,
                    "labels":[]
                }

                for pairs in sample['dissonance_pairs']:

                    du1_sent = dus[pairs['du1']]
                    du2_sent = dus[pairs['du2']]
                    before_du1= " ".join(dus[:pairs['du1']])
                    after_du2 = " ".join(dus[pairs['du2']:])
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['list_pairs'].append([du1_sent,du2_sent])
                    item['before_after_sep'].append(before_du1+" <s> "+after_du2)
                    item['labels'].append(disso_label)

                dataset.append(item)

            return DissonanceDatasetwithbeforeaftersep(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "id": _inst["id"],
            "list_pairs": _inst["list_pairs"],
            "tweet": _inst["tweet"],
            "labels": _inst['labels'],
            "before_after_sep": _inst["before_after_sep"],
        }

        return item

class DissonanceWholeTweetWithSep(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = "</s>".join(dus)

                item = {
                    "id": sample['message_id'],
                    'du_pairs_index':None,
                    "tweet": full_message,
                    "labels":None
                }

                for pairs in sample['dissonance_pairs']:

                    du_pairs_index = [pairs['du1'],pairs['du2']]
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['du_pairs_index']=du_pairs_index
                    item['labels']=disso_label
                    dataset.append(item)

            return DissonanceWholeTweetWithSep(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "id": _inst["id"],
            "du_pairs_index": _inst["du_pairs_index"],
            "tweet": _inst["tweet"],
            "labels": _inst['labels']
        }

        return item

class kialoDisagreeement(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:

                full_message = sample['belief1']+" </s> "+sample['belief2']
                disso_label = {"AGREE": 0, "DISAGREE": 1, "N/A": 2}[sample['label']]
                item = {
                    "tweet": full_message,
                    "labels":disso_label
                }
                dataset.append(item)

            return kialoDisagreeement(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "tweet": _inst["tweet"],
            "labels": _inst['labels']
        }

        return item

class DissonanceDatasetBeforeafterWhole(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = " ".join(dus)

                item = {
                    "tweet":None,
                    "labels":None
                }

                for pairs in sample['dissonance_pairs']:

                    du1_sent = dus[pairs['du1']]
                    du2_sent = dus[pairs['du2']]
                    before_du1= " ".join(dus[:pairs['du1']])
                    after_du2 = " ".join(dus[pairs['du2']+1:])
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['tweet']=full_message+" </s> "+du1_sent+" </s> "+du2_sent+" </s> "+before_du1+" </s> "+after_du2
                    item['labels']=disso_label
                    dataset.append(item)

            return DissonanceDatasetBeforeafterWhole(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {

            "tweet": _inst["tweet"],
            "labels": _inst['labels']
        }

        return item

class DissonanceDatasetTweetDus(torch.utils.data.Dataset):
    @staticmethod
    def from_files(file, two_class=False):
            dataset = list()
            local_dataset = json.load(open(file))
            logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")


            for sample in local_dataset:
                dus=sample['message'].split("-->>")
                dus= [sent.strip() for sent in  dus]

                full_message = " ".join(dus)

                item = {
                    "id": sample['message_id'],
                    'list_pairs':[],
                    "tweet": full_message,
                    "topic": "topic",
                    "labels":[]
                }

                for pairs in sample['dissonance_pairs']:

                    du1_sent = dus[pairs['du1']]
                    du2_sent = dus[pairs['du2']]
                    disso_label = {"C": 0, "D": 1, "N": 2}[pairs['disso_label']]
                    item['list_pairs'].append(full_message+" </s> "+du1_sent+" </s> "+du2_sent)
                    item['labels'].append(disso_label)

                dataset.append(item)

            return DissonanceDatasetTweetDus(dataset, two_class)

    def __init__(self, dataset, two_class=False):
        self.dataset = np.array(dataset)
        self.two_class = two_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _inst = self.dataset[idx]
        item = {
            "id": _inst["id"],
            "topic": _inst["topic"],
            "list_pairs": _inst["list_pairs"],
            "tweet": _inst["tweet"],
            "labels": _inst['labels']
        }

        return item

# class pdtbDisagreeement(torch.utils.data.Dataset):
#     @staticmethod
#     def from_files(file, two_class=False):
#             dataset = list()
#             local_dataset = json.load(open(file))
#             logging.info(f"Loaded: {file} ({len(local_dataset)} instances).")
#
#
#             for sample in local_dataset:
#
#                 full_message = sample['belief1']+" </s> "+sample['belief2']
#                 disso_label = {"AGREE": 0, "DISAGREE": 1}[sample['label']]
#                 item = {
#                     "tweet": full_message,
#                     "labels":disso_label
#                 }
#                 dataset.append(item)
#
#             return pdtbDisagreeement(dataset, two_class)
#
#     def __init__(self, dataset, two_class=False):
#         self.dataset = np.array(dataset)
#         self.two_class = two_class
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         _inst = self.dataset[idx]
#         item = {
#             "tweet": _inst["tweet"],
#             "labels": _inst['labels']
#         }
#
#         return item