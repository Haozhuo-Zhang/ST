import torch
import torchvision.models as models
import os


class Vgg19(torch.nn.Module):
    """
    style: 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'
    content: 'conv4_2'
    """

    def __init__(self, requires_grad=False, use_relu=True):
        super().__init__()
        if os.path.exists("../vgg19/vgg19-dcbb9e9d.pth"):
            vgg19_model = models.vgg19(weights=None)
            vgg19_model.load_state_dict(torch.load("../vgg19/vgg19-dcbb9e9d.pth"))
            pretrained_vgg19 = vgg19_model.features
        else:
            pretrained_vgg19 = models.vgg19(
                weights=models.VGG19_Weights.DEFAULT
            ).features

        if use_relu:  # use relu or as in original paper conv layers
            self.layer_names = [
                "relu1_1",
                "relu2_1",
                "relu3_1",
                "relu4_1",
                "conv4_2",
                "relu5_1",
            ]
            self.offset = 1
        else:
            self.layer_names = [
                "conv1_1",
                "conv2_1",
                "conv3_1",
                "conv4_1",
                "conv4_2",
                "conv5_1",
            ]
            self.offset = 0
        self.content_index = 4  # conv4_2

        # all layers except conv4_2
        self.style_index = [0, 1, 2, 3, 5]

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()

        for x in range(1 + self.offset):
            self.slice1.add_module(str(x), pretrained_vgg19[x])
        for x in range(1 + self.offset, 6 + self.offset):
            self.slice2.add_module(str(x), pretrained_vgg19[x])
        for x in range(6 + self.offset, 11 + self.offset):
            self.slice3.add_module(str(x), pretrained_vgg19[x])
        for x in range(11 + self.offset, 20 + self.offset):
            self.slice4.add_module(str(x), pretrained_vgg19[x])
        for x in range(20 + self.offset, 22):
            self.slice5.add_module(str(x), pretrained_vgg19[x])
        for x in range(22, 29 + self.offset):
            self.slice6.add_module(str(x), pretrained_vgg19[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        out = (layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out


def init_model(device):
    model = Vgg19(requires_grad=False, use_relu=True)
    content_index = model.content_index  # 4
    style_indices = model.style_index  # 0, 1, 2, 3, 5
    model.to(device).eval()

    return model, content_index, style_indices
