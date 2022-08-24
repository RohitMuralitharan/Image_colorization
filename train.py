import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import cv2
from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.optim as optim
from model import define_G, define_D, GANLoss, get_scheduler, update_learning_rate

# reading images
images = []
for filename in os.listdir("landscape_images"):
    img = cv2.imread(os.path.join("landscape_images",filename))
    if img is not None:
        images.append(img)

# converting the images to gray_scale
input_transform = T.Compose([T.ToTensor(),
                             T.Resize(size=(256,256)),
                             T.Grayscale(),
                             T.Normalize((0.5), (0.5))
                             ])
input_img = []
for i in images:
    input_img.append(input_transform(i))    
len(input_img)

# converting the torch into pic
img = []
for i in range(len(input_img)):
    img.append(ToPILImage()(input_img[i]))

# converting the grayscale image into array of height 3
input_img = []
image = []
for i in range(len(img)):
    image = img[i].convert('RGB')
    image = image.resize((286, 286), Image.BICUBIC)
    image = transforms.ToTensor()(image)
    w_offset = random.randint(0, max(0, 286 - 256 - 1))
    h_offset = random.randint(0, max(0, 286 - 256 - 1))
    image = image[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    idx = [j for j in range(image.size(2) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    image = image.index_select(2, idx)
    input_img.append(image)

# reading the true image
target_transform = T.Compose([T.ToTensor(),
                              T.Resize(size=(256,256)),
                              T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
target_img = []
for i in images:
    target_img.append(target_transform(i))    
len(target_img)

# test-train split
X_train, X_test, y_train, y_test = train_test_split(input_img, target_img, test_size=0.2, random_state=42)

train_set = X_train,y_train
test_set = X_test, y_test

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
args = parser.parse_args(args=[])

training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

device = torch.device("cuda:0" if args.cuda else "cpu")

print('===> Building models')
net_g = define_G(args.input_nc, args.output_nc, args.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(args.input_nc + args.output_nc, args.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, args)
net_d_scheduler = get_scheduler(optimizer_d, args)

for epoch in range(args.epoch_count, args.niter + args.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
       
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
       
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * args.lamb
       
        loss_g = loss_g_gan + loss_g_l1
       
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * np.math.log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", "Model")):
            os.mkdir(os.path.join("checkpoint", "Model"))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format("Model", epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format("Model", epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + "Model"))
