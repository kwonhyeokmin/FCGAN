import os.path

import cv2
import torch
from datasets.unpaireddataset import UnpairedDataset
from torch.utils.data import DataLoader
from config import cfg
from PIL import Image
import numpy as np


if __name__ == '__main__':
    # Select device (gpu | cpu)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # ********************
    # 1. Load datasets
    # ********************
    dataset = UnpairedDataset('test')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_thread,
                            pin_memory=True)

    for i, data in enumerate(dataloader):
        A_tensor = data['A'].to(device).permute(0, -1, 1, 2)
        B, C, H, W = A_tensor.shape
        fl = 5
        A_fre_tensor = torch.fft.fft2(A_tensor, norm='backward')
        print(A_tensor.shape)
        print(A_fre_tensor.shape)
        print()
        A_fre_low_tensor = A_fre_tensor[:,:,:,:fl]
        A_fre_high_tensor = A_fre_tensor[:,:,:,fl:]
        for t, _type in zip([A_fre_low_tensor, A_fre_high_tensor], ['low', 'high']):
            A_out = torch.fft.irfft2(t, s=(H, W), norm='backward')
            # save
            a = (A_out[0].cpu().detach().numpy()).astype(np.uint8).transpose(1, 2, 0)
            cv2.imwrite(os.path.join(cfg.vis_dir, f'test_{_type}.jpg'), a)
        if i > 50:
            break