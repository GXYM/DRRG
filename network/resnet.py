import torch.nn as nn
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet(nn.Module):
    def __init__(self, name="resnet50", pretrain=True):
        super().__init__()

        if name == "resnet50":
            base_net = resnet.resnet50(pretrained=False)
        elif name == "resnet101":
            base_net = resnet.resnet101(pretrained=False)
        else:
            print(" base model is not support !")

        if pretrain:
            print("load the {} weight from ./cache".format(name))
            base_net.load_state_dict(model_zoo.load_url(model_urls["resnet50"], model_dir="./cache"))
        # print(base_net)
        self.stage1 = nn.Sequential(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool
        )
        self.stage2 = base_net.layer1
        self.stage3 = base_net.layer2
        self.stage4 = base_net.layer3
        self.stage5 = base_net.layer4
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)

        # up2 --> 1/2
        C1 = self.up2(C1)

        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 512, 512))
    net = ResNet()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
