import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class mnistEfficient(torch.nn.Module):
    def __init__(self) -> None:
        super(mnistEfficient, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6')
        self.classifier = torch.nn.Linear(1000, 26)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        x = F.silu(self.backbone(x))
        x = self.classifier(x)

        return x