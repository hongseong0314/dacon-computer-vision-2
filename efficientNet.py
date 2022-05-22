import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class mnistEfficient(torch.nn.Module):
    def __init__(self) -> None:
        super(mnistEfficient, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6')

        self.backbone._fc = torch.nn.Linear(1000, 512)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.activation = self.backbone._swish
        self.classifier = torch.nn.Linear(512, 26)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.classifier(x)

        return x