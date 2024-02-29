from torch import nn
import timm
import torch

class HiddenLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(HiddenLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class CXRModel(nn.Module):

    def __init__(self):
        """_summary_
        https://github.com/circ-ml/CXR-LC
        """
        super(CXRModel, self).__init__()

        # Grayscale & 1 class regression
        self.img_model = timm.create_model("inception_v4", in_chans=1, num_classes=32, pretrained=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.layer1 = HiddenLayer(2, 32)
        self.layer2 = HiddenLayer(32, 32)
        self.layer3 = HiddenLayer(64, 32)
        self.layer4 = HiddenLayer(32, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, tab, img):
        tab = self.layer1(tab)
        tab = self.layer2(tab)
        img = self.img_model(img)
        img = self.relu(img)
        img = self.bn(img)
        x = torch.cat([tab, img], dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return nn.functional.softplus(x)

if __name__ == "__main__":
    model = CXRModel()
    print(model)