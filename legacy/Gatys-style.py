import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

writer = torch.utils.tensorboard.SummaryWriter(f"./runs/Gatys-style")

d2l.set_figsize()
d2l.Image.MAX_IMAGE_PIXELS = None
content_img = d2l.Image.open("./cbd.jpg")
style_img = d2l.Image.open("./Starry_Night.jpg")

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )
    return transforms(img).unsqueeze(0)


def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return img.permute(2, 0, 1)


pretrained_net = torchvision.models.vgg19(pretrained=False)
pretrained_net.load_state_dict(
    torch.load("/data/ckpt/zhanghaozhuo/Style Transfer/vgg19-dcbb9e9d.pth")
)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

net = nn.Sequential(
    *[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)]
)


def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    return 0.5 * (
        torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean()
        + torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()
    )


content_weight, style_weight, tv_weight = 1, 1e4, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [
        content_loss(Y_hat, Y) * content_weight
        for Y_hat, Y in zip(contents_Y_hat, contents_Y)
    ]
    styles_l = [
        style_loss(Y_hat, Y) * style_weight
        for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)
    ]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram
        )
        l.backward()
        trainer.step()
        scheduler.step()

        cl = float(sum(contents_l))
        sl = float(sum(styles_l))
        tl = float(tv_l)
        results = open(
            "/data/ckpt/zhanghaozhuo/Style Transfer/Gatys-style-logs.txt", "a"
        )
        results.write(f"contents_l: {cl}, styles_l: {sl}, tv_l: {tl}\n")
        results.close()
        writer.add_scalar("contents_l", cl, epoch)
        writer.add_scalar("styles_l", sl, epoch)
        writer.add_scalar("tv_l", tl, epoch)

        if (epoch + 1) % 50 == 0:
            print(f"epoch: {epoch+1}, contents_l: {cl}, styles_l: {sl}, tv_l: {tl}\n")
            img = postprocess(X)
            torchvision.utils.save_image(
                img,
                f"/data/ckpt/zhanghaozhuo/Style Transfer/Gatys-style-outputs/epoch_{epoch + 1}.jpg",
            )

    writer.flush()
    return X


shape_reversed = tuple(map(int, tuple(map(lambda x: x / 2, content_img.size))))
image_shape = (shape_reversed[1], shape_reversed[0])
device = torch.device("cuda:2")  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
