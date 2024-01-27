import argparse
import torch
import torch.optim as optim
import torchvision
import util
import models


def loadargs():
    data_dir = "./images"
    content_img_dir = f"{data_dir}/content/"
    style_img_dir = f"{data_dir}/style/"
    output_img_dir = f"{data_dir}/output/"
    img_format = ".jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content", type=str, default="CBD.jpg", help="content image, default CBD"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="StarryNight.jpg",
        help="style image, default StarryNight",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["random", "content", "style"],
        default="content",
        help="init method of output image, default content",
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1,
        help="weight of content loss, default 1",
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=1e15,
        help="weight of style loss, default 1e15",
    )
    parser.add_argument(
        "--tv_weight", type=float, default=1, help="weight of tv loss, default 1"
    )
    parser.add_argument(
        "--saving_freq",
        type=int,
        default=100,
        help="saving frequency of output images, default 100",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[-1, -1],
        help='shape of output image, default "-1 -1" for shape of content image / 2, or "width -1" to set width and let height scale from content image, or "width height" for custom setting',
    )
    parser.add_argument("--lapstyle", action="store_true", help="use lapstyle")
    parser.add_argument(
        "--lap_weight",
        type=float,
        default=1e3,
        help="weight of laplacian loss, default 1e3",
    )
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config["content_img_dir"] = content_img_dir
    config["style_img_dir"] = style_img_dir
    config["output_img_dir"] = output_img_dir
    config["img_format"] = img_format
    return config


# load model
if __name__ == "__main__":
    print("Loading arguments...")
    config = loadargs()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Preparing imagings and model...")
    print(f"content image: {config['content']}\nstyle image: {config['style']}")
    content_img, style_img, out_img = util.loadimgs(config, device, False)

    print("Resnet model used: Resnet50")
    model, content_index, style_indices = models.init_resnet_model(device)

    content_weight, style_weight, tv_weight, lap_weight = (
        config["content_weight"],
        config["style_weight"],
        config["tv_weight"],
        config["lap_weight"],
    )
    if not config["lapstyle"]:
        lap_weight = 0

    content_feature = model(content_img)
    style_feature = model(style_img)

    target_content_representation = content_feature[content_index]
    target_style_representation = util.get_style_rep(style_feature, style_indices)

    lapmodel = models.LapPyramid(device)
    target_content_lap_representation = lapmodel(content_img)

    savedir = util.prepare_savedir(config, "resnet")
    optimizer = optim.LBFGS([out_img])

    it = 0
    iterations = 1000

    def closure():
        global it
        optimizer.zero_grad()
        gen_feature = model(out_img)

        content_loss = torch.nn.MSELoss(reduction="mean")(
            gen_feature[content_index], target_content_representation
        )

        style_loss = 0.0
        gen_style_rep = util.get_style_rep(gen_feature, style_indices)
        for gram_gt, gram_hat in zip(target_style_representation, gen_style_rep):
            style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
        style_loss /= len(target_style_representation)

        tv_loss = util.total_variation(out_img)

        lap_loss = util.get_laplacian_loss(
            lapmodel, out_img, target_content_lap_representation
        )

        total_loss = (
            content_weight * content_loss
            + style_weight * style_loss
            + tv_weight * tv_loss
            + lap_weight * lap_loss
        )
        total_loss.backward()

        if it % config["saving_freq"] == 0:
            with torch.no_grad():
                if config["lapstyle"]:
                    print(
                        f"iteration: {it:04}, total loss={total_loss.item()}, content_loss={content_weight * content_loss.item()}, style loss={style_weight * style_loss.item()}, tv loss={tv_weight * tv_loss.item()}, lap loss={lap_weight * lap_loss.item()}"
                    )
                else:
                    print(
                        f"iteration: {it:04}, total loss={total_loss.item()}, content_loss={content_weight * content_loss.item()}, style loss={style_weight * style_loss.item()}, tv loss={tv_weight * tv_loss.item()}"
                    )
        if it % config["saving_freq"] == 0:
            torchvision.utils.save_image(
                util.postprocess(out_img, False), f"{savedir}/iter{it:04}.jpg"
            )
        it += 1
        return total_loss

    while it <= iterations:
        optimizer.step(closure)
