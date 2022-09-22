import os
import shutil
import sys
import time
import torch
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloder import CustomImageFolderDataset
from models.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim, lr_schedule_cosdecay
from torchvision.models import vgg16
import torchvision.utils as utils
import math
import torchvision.transforms as T
from tqdm import tqdm
from simulator_new import Simulator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    
    train_batch_size, test_batch_size = 8, 32
    num_epochs = 50
    all_T = 100000
    save_dir = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112/checkpoint"

    if not os.path.isfile(save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)


    net = get_model()
    net = torch.nn.DataParallel(net)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    dataset = CustomImageFolderDataset(root = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112/gt",
                                       transform=T.Compose([T.ToTensor(),T.RandomCrop(112)]),
                                       target_transform=None)
    params = {'batch_size': train_batch_size,
              'shuffle': True,
              'num_workers': 8}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    print("DATALOADER DONE!")
    
    turb_params = {
                'img_size': (112,112),
                'D':0.071,        # Apeture diameter
                'r0':0.071,      # Fried parameter 
                'L':100,       # Propogation distance
                'thre':0.02,   # Used to suppress small values in the tilt correlation matrix. Increase 
                                # this threshold if the pixel displacement appears to be scattering
                'adj':1,        # Adjusting factor of delta0 for tilt matrix
                'wavelength':0.500e-6,
                'corr':-0.05,    # Correlation strength for PSF without tilt. suggested range: (-1 ~ -0.01)
                'zer_scale':1   # manually adjust zernike coefficients of the PSF without tilt.
            }
    simulator = torch.nn.DataParallel(Simulator(turb_params, data_path="utils")).cuda()
    print("===> Training Start ...")
    for epoch in range(num_epochs):
        print("====> Training Eoch [{}/{}]".format(epoch, num_epochs))
        start_time = time.time()
        psnr_list = []
        # --- Save the network parameters --- #
        torch.save(net.state_dict(), '{}/checkpoint_{}.pth'.format(save_dir, epoch))

        for batch_id, train_data in tqdm(enumerate(dataloader)):
            if batch_id > 5000:
                break
            step_num = batch_id + epoch * 5000 + 1
            lr=lr_schedule_cosdecay(step_num,all_T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            turb, gt, noise_loaded, target = train_data
            turb = turb.cuda()
            gt = gt.cuda()
            noise = (noise_loaded[0].squeeze(1).cuda(), noise_loaded[1].squeeze(1).cuda())

            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            _, J, T, I = net(turb)
            noise, _, _, sim_I = simulator(J, noise)

            Rec_Loss1 = F.smooth_l1_loss(J, gt)
            Rec_Loss2 = F.smooth_l1_loss(I, turb)
            Rec_Loss3 = F.smooth_l1_loss(sim_I, turb)

            loss = Rec_Loss1 + Rec_Loss2 + Rec_Loss3
            loss.backward()
            optimizer.step()

            # --- To calculate average PSNR --- #
            psnr_list.extend(to_psnr(J, gt))

            if not (batch_id % 100):
                print('Epoch: {}, Iteration: {}, Loss: {:.3f}, Rec_Loss1: {:.3f}, Rec_loss2: {:.3f}, Rec_loss3: {:.3f}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2, Rec_Loss3))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)
        print("Train PSNR : {:.3f}".format(train_psnr))
    
    
    
