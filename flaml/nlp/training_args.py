from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class TuneTrainingArguments(TrainingArguments):
    """
     TuneTrainingArguments overrides the default TrainingArguments class**.

     Using :class:`~transformers.HfArgumentParser` we can turn this class into argparse arguments to be able to specify
     them on the command line.

     Parameters:
         hidden_dropout_prob:
            hidden dropout probability
         attention_probs_dropout_prob:
            attention dropout probability
    """

    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "dropout ratio"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "dropout ratio"})

