from torchvision.models import resnet50
from torchinfo import summary
import torch
import torch.nn.functional as F

class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = resnet50() 
        self.classifier = torch.nn.Linear(1000, 26)
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        # feature extract
        x = F.relu(self.resnet(x))
        # model classifier
        x = self.classifier(x)
        return x