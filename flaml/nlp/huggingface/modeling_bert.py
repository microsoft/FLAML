from torch import nn

class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):# take <s> token (equiv. to [CLS])
        x = self.classifier(features)
        return x