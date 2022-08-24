import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
args = parser.parse_args(args=[])

device = torch.device("cuda:0" if args.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format("Model", args.nepochs)

net_g = torch.load(model_path).to(device)

image_dir = "dataset/test/"

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", "Result")):
        os.makedirs(os.path.join("result", "Result"))
    save_img(out_img, "result/{}/{}".format("Result", image_name))