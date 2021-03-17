from transformers import TrainingArguments
from dataclasses import dataclass, field

@dataclass
class TuneTrainingArguments(TrainingArguments):
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "dropout ratio"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "dropout ratio"})

