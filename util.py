import torch
from torchvision import transforms
import os

rgb_mean = torch.tensor([123.675, 116.28, 103.53])
rgb_std = torch.tensor([1, 1, 1])


def preprocess(img, image_shape, device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )
    return transform(img).to(device).unsqueeze(0)  # (1, C, H, W)


def postprocess(img):
    img = img[0].to("cpu").detach()  # cpu
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 255)
    return img.permute(2, 0, 1) / 255


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


def prepare_savedir(config):
    savedir = f"{config['output_img_dir']}{config['content'].split('.')[0]}_{config['style'].split('.')[0]}"
    os.makedirs(savedir, exist_ok=True)
    return savedir
