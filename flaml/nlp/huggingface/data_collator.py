from dataclasses import dataclass
from transformers.data.data_collator import (
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorMixin,
)
from typing import Optional

from flaml.data import MULTICHOICECLASSIFICATION, TOKENCLASSIFICATION


@dataclass
class DataCollatorForMultipleChoiceClassification(DataCollatorWithPadding):
    def __call__(self, features):
        from itertools import chain
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature.pop(label_name) for feature in features]
            if label_name in features[0]
            else None
        )

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))
        batch = super(DataCollatorForMultipleChoiceClassification, self).__call__(
            flattened_features
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


@dataclass
class DataCollatorForAuto(
    DataCollatorForTokenClassification,
    DataCollatorForMultipleChoiceClassification,
):
    task: Optional[str] = None

    def __call__(self, features):
        if self.task == MULTICHOICECLASSIFICATION:
            return super(DataCollatorMixin, self).__call__(
                features
            )  # Will call DataCollatorForMultipleChoiceClassification
        elif self.task == TOKENCLASSIFICATION:
            return super().torch_call(
                features
            )  # Will call DataCollatorForTokenClassification
