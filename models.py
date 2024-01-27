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
        if os.path.exists("./vgg19/vgg19.pth"):
            vgg19_model = models.vgg19(weights=None)
            vgg19_model.load_state_dict(torch.load("./vgg19/vgg19.pth"))
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
        self.style_indices = [0, 1, 2, 3, 5]

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


def init_vgg_model(device):
    model = Vgg19(requires_grad=False, use_relu=True)
    content_index = model.content_index  # 4
    style_indices = model.style_indices  # 0, 1, 2, 3, 5
    model.to(device).eval()

    return model, content_index, style_indices


class LapPyramid(torch.nn.Module):
    def __init__(self, device, n_levels=5):
        super(LapPyramid, self).__init__()
        self.n_levels = n_levels
        laplacian_kernel = (
            torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.laplacian_kernel = laplacian_kernel.repeat(1, 3, 1, 1).to(device)
        self.downsample = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        # x: [b, c, h, w]
        laplacians = []
        for i in range(self.n_levels):
            laplacian = torch.nn.functional.conv2d(x, self.laplacian_kernel, padding=1)
            laplacians.append(laplacian)
            x = self.downsample(x)
        return laplacians


class Resnet50(torch.nn.Module):
    def __init__(self, alpha=0.001, requires_grad=False):
        super().__init__()
        if os.path.exists("./resnet50/resnet50.pth"):
            self.resnet50 = models.resnet50(weights=None)
            self.resnet50.load_state_dict(torch.load("./resnet50/resnet50.pth"))
        else:
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if not requires_grad:
            for param in self.resnet50.parameters():
                param.requires_grad_(False)
        # self.layer1 = ["conv1_0", "conv1_1", "conv1_2"]
        # self.layer2 = ["conv2_0", "conv2_1", "conv2_2", "conv2_3"]
        # self.layer3 = [
        #     "conv3_0",
        #     "conv3_1",
        #     "conv3_2",
        #     "conv3_3",
        #     "conv3_4",
        #     "conv3_5",
        # ]
        # self.layer4 = ["conv4_0", "conv4_1", "conv4_2"]
        # self.style_indices = ["conv0_0", "conv1_2", "conv2_3", "conv3_5", "conv4_2"]
        # self.content_index = "conv3_5"
        self.alpha = alpha
        self.style_indices = [0, 1, 2, 3, 4]
        self.content_index = 3

    def forward(self, x):
        features = []
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        features.append(x)  # 0
        x = self.resnet50.maxpool(x)

        for layer in self.resnet50.layer1:
            x = layer(x)
        features.append(x)  # 1

        for layer in self.resnet50.layer2:
            x = layer(x)
        features.append(x)  # 2

        for layer in self.resnet50.layer3:
            x = layer(x)
        features.append(x * self.alpha)  # 3

        for layer in self.resnet50.layer4:
            x = layer(x)
        features.append(x * self.alpha)  # 4
        return features


def init_resnet_model(device):
    model = Resnet50()
    content_index = model.content_index
    style_indices = model.style_indices
    model.to(device).eval()

    return model, content_index, style_indices
