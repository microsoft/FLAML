# -*- coding: utf-8 -*- 
# @Time : 12/22/2021 5:22 PM 
# @Author : Ethan
# @File : data_collator.py 
# @Description :
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorWithPadding

@dataclass
class DataCollatorForAuto(DataCollatorWithPadding):
    def __call__(self, features):
        from itertools import chain
        import torch
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))
        batch = super(DataCollatorForAuto, self).__call__(flattened_features)
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
