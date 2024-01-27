import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
rgb_mean_scale = torch.tensor([123.675, 116.28, 103.53])
rgb_std_scale = torch.tensor([1, 1, 1])


def preprocess(img, image_shape, device, scale=True) -> torch.Tensor:
    if scale:
        transform = transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                transforms.Normalize(mean=rgb_mean_scale, std=rgb_std_scale),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=rgb_mean, std=rgb_std),
            ]
        )
    return transform(img).to(device).unsqueeze(0)  # (1, C, H, W)


def postprocess(img, scale=True):
    img = img[0].to("cpu").detach()  # cpu
    if scale:
        img = torch.clamp(img.permute(1, 2, 0) * rgb_std_scale + rgb_mean_scale, 0, 255)
        return img.permute(2, 0, 1) / 255
    else:
        img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
        return img.permute(2, 0, 1)


def loadimgs(config, device, scale=True):
    content_img = Image.open(config["content_img_dir"] + config["content"])
    style_img = Image.open(config["style_img_dir"] + config["style"])
    if config["shape"] == [-1, -1]:
        image_shape = (content_img.size[1] // 2, content_img.size[0] // 2)
    elif config["shape"][1] == -1:
        image_shape = (
            config["shape"][0],
            config["shape"][0] * content_img.size[0] // content_img.size[1],
        )
    else:
        image_shape = config["shape"]

    content_img = preprocess(content_img, image_shape, device, scale)

    style_img = preprocess(style_img, image_shape, device, scale)

    if config["init"] == "random":
        out_img = np.random.normal(loc=0, scale=255 / 2, size=content_img.shape).astype(
            np.float32
        )
        out_img = torch.from_numpy(out_img).to(device)
    elif config["init"] == "content":
        out_img = content_img.clone()
    elif config["init"] == "style":
        out_img = style_img.clone()
    out_img.requires_grad = True

    return content_img, style_img, out_img


def prepare_savedir(config, path=""):
    savedir = f"{config['output_img_dir']}{path}_{config['content'].split('.')[0]}2{config['style'].split('.')[0]}"
    if config["lapstyle"]:
        savedir += "/lapstyle"
    else:
        savedir += "/gatys"
    os.makedirs(savedir, exist_ok=True)
    return savedir


def gram_matrix(x, should_normalize=True):
    # batch
    (b, c, h, w) = x.size()
    features = x.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= c * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
        torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    )


def get_content_rep(img_feature, content_index):
    return img_feature[content_index].squeeze(axis=0)  # (512, h/8, w/8)


def get_style_rep(img_feature, style_indices):
    return [gram_matrix(x) for cnt, x in enumerate(img_feature) if cnt in style_indices]


def get_gatys_loss(
    vggmodel,
    generating_img_tensor,
    target_content_rep,
    content_index,
    target_style_rep,
    style_indices,
    content_weight,
    style_weight,
    tv_weight,
):
    gen_feature = vggmodel(generating_img_tensor)
    gen_content_rep = get_content_rep(gen_feature, content_index)
    gen_style_rep = get_style_rep(gen_feature, style_indices)

    content_loss = torch.nn.MSELoss(reduction="mean")(
        target_content_rep, gen_content_rep
    )

    style_loss = 0.0
    for gram_gt, gram_hat in zip(target_style_rep, gen_style_rep):
        style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_rep)

    tv_loss = total_variation(generating_img_tensor)

    total_loss = (
        content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
    )

    return total_loss, content_loss, style_loss, tv_loss


def get_laplacian_loss(lapmodel, generating_img_tensor, target_content_lap_rep):
    lapstyle_weights = [16, 8, 4, 2, 1]
    gen_laplacians = lapmodel(generating_img_tensor)

    lap_loss = 0.0
    for i in range(len(lapstyle_weights)):
        lap_loss += lapstyle_weights[i] * torch.nn.MSELoss(reduction="mean")(
            gen_laplacians[i], target_content_lap_rep[i]
        )
    return lap_loss


def get_lapstyle_loss(
    vggmodel,
    generating_img_tensor,
    target_content_rep,
    content_index,
    target_style_rep,
    style_indices,
    content_weight,
    style_weight,
    tv_weight,
    lapmodel,
    target_content_lap_rep,
    lap_weight,
):
    total_loss, content_loss, style_loss, tv_loss = get_gatys_loss(
        vggmodel,
        generating_img_tensor,
        target_content_rep,
        content_index,
        target_style_rep,
        style_indices,
        content_weight,
        style_weight,
        tv_weight,
    )
    lap_loss = get_laplacian_loss(
        lapmodel, generating_img_tensor, target_content_lap_rep
    )
    total_loss += lap_loss * lap_weight
    return total_loss, content_loss, style_loss, tv_loss, lap_loss
