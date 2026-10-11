import os

import pickle
import time
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from config import config
from models.basic_modules import A_CDP, Poisson_noise_torch
from models.diffpr_net import DiffusionModel

def save_image(path, image_name, x_hat):
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f"{path}/{image_name}.png", x_hat)  

def imread_CS_py(Iorg):
    block_size = config.para.patch_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def test(network, val, save_img=config.para.save, device=config.para.device):
    recon_root = f"./{config.para.reconstruct_results}/"
   
    datasets = ['UNT6', 'NT6']

    with torch.no_grad():
        for one_dataset in datasets:
            print(one_dataset + " reconstruction start")
            test_dataset_path = f"{config.para.dataset_root}/{one_dataset}"
            
            sum_psnr, sum_ssim, sum_time = 0., 0., 0.
            for _, _, images in os.walk(f"{test_dataset_path}/"):
                for one_image in tqdm(images, desc='Now testing:'):
                    name_image = one_image.split('.')[0]
                    Img = cv2.imread(f"{test_dataset_path}/{one_image}", flags=1)
                    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()
                    Iorg_y = Img_yuv[:, :, 0]
                    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                    Img_output = Ipad / 255.

                    batch_x = torch.from_numpy(Img_output)
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_x = batch_x.to(config.para.device)
                    batch_x = batch_x.unsqueeze(0).unsqueeze(0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=3),
                                        dim=0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=2),
                                        dim=0)

                    Mask_data_Name = './%s/mask_%d_%d_test.p' % (config.para.matrix_dir, config.para.rate, config.para.patch_size)
                    if os.path.exists(Mask_data_Name):
                        Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
                   
                    # uniform mask generation
                    # generate complex mask, 1j == i
                    Mask_data = torch.exp(
                        1j * 2 * torch.pi * torch.rand(1, config.para.rate, config.para.patch_size, config.para.patch_size))
                    pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                    
                    mask = Mask_data.to(device)
                    b = Poisson_noise_torch(A_CDP(batch_x, SamplingRate=config.para.rate, mask=mask),
                                            alpha=config.para.alpha)
                    initial_data = torch.ones_like(batch_x)
                    time_start = time.time()
                    x_output = network(initial_data, mask, b)
                    time_end = time.time()
                    time_item = time_end - time_start
                    sum_time += time_item

                    x_output = torch.cat(
                        torch.split(x_output, split_size_or_sections=1 * col_new // config.para.patch_size, dim=0),
                        dim=2)
                    x_output = torch.cat(torch.split(x_output, split_size_or_sections=1, dim=0), dim=3)
                    x_output = x_output.squeeze(0).squeeze(0)
                    Prediction_value = x_output.cpu().data.numpy()

                    X_rec = Prediction_value[:row, :col]  

                    X_rec = np.clip(X_rec, 0, 1) * 255.

                    rec_PSNR = PSNR(X_rec, Iorg.astype(np.float64), data_range=255)
                    rec_SSIM = SSIM(X_rec, Iorg.astype(np.float64), data_range=255)

                    del x_output
                    sum_psnr += rec_PSNR
                    sum_ssim += rec_SSIM

                    if save_img:
                        Img_rec_yuv[:, :, 0] = X_rec
                        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                        save_image(f"{recon_root}/{config.para.rate}/{one_dataset}",
                                   f'{name_image}_noise{config.para.alpha}_{round(rec_PSNR,2)}_{round(rec_SSIM,3)}', im_rec_rgb)
                    print(name_image, 'PSNR:', rec_PSNR, 'SSIM:', rec_SSIM, 'time:', time_item)
                print('Average results: ', one_dataset, sum_psnr / len(images), sum_ssim / len(images), sum_time / len(images))
                               
    return            



def main():
    config.para.my_state_dict = f"{config.para.save_dir}/{str(config.para.rate)}/{str(config.para.rate)}_state_dict.pth"
    os.makedirs(config.para.matrix_dir, exist_ok=True)

    my_state_dict = config.para.my_state_dict
    device = config.para.device

    betas = torch.linspace(1e-4, 0.02, config.para.time_steps, device=device)

    net = DiffusionModel(T=config.para.time_steps, betas=betas, rate=config.para.rate).eval().to(device)
    if os.path.exists(my_state_dict):
        if torch.cuda.is_available():
            trained_model = torch.load(my_state_dict, map_location=device)
        else:
            raise Exception(f"No GPU.")
        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Missing trained model of rate {config.para.rate}, {my_state_dict}.")
    
    test(net, val=False, save_img=config.para.save)



if __name__ == '__main__':
    main()
