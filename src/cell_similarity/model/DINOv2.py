import torch
import torch.nn as nn

class DINOv2(nn.module):
    def __init__(
            self,
            model_name,
    ):
        super().__init__()
        self.model = torch.hub.load(model_name)

    def forward(self, *args, **kargs):
        logits = self.model(*args, **kargs)

        return logits
