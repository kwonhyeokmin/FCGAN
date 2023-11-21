import torch
from tqdm import tqdm
from config import cfg
from datasets.unpaireddataset import UnpairedDataset
from torch.utils.data import DataLoader
from models.cycle_gan import CycleGAN
from models.fcvgan import FCVGAN
from PIL import Image
import numpy as np
import os
import os.path as osp
import argparse
from common.utils.dir_utils import update_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--model',
                        help='choice model for training',
                        choices=['cyclegan', 'fcvgan'],
                        default='fcvgan',
                        type=str)
    args = parser.parse_args()

    return args

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

if __name__ == '__main__':
    args = parse_args()
    update_config(config_file=args.cfg, config=cfg)
    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ********************
    # 1. Load datasets
    # ********************
    dataset = UnpairedDataset('test')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_thread,
                            pin_memory=True)

    # ****************
    # 2. Load model
    # ****************
    network = None
    if args.model == 'cyclegan':
        network = CycleGAN
    elif args.model == 'fcvgan':
        network = FCVGAN
    else:
        raise AssertionError('Invalid model name')
    model = network()
    model.setup()
    model.eval()

    # ****************
    # 3. Test
    # ****************
    unnorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    for i, data in enumerate(tqdm(dataloader)):
        model.set_input(data)
        model.test()
        for b in range(len(data['A_paths'])):
            image_numpy = unnorm(model.fake_B[b]).cpu().float().numpy()
            # image_numpy = model.i9_b[b].cpu().float().numpy()
            # image_numpy = model.a10_b[b].cpu().float().numpy()
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

            image_pil = Image.fromarray(image_numpy.astype(np.uint8))
            image_pil.save(osp.join(cfg.vis_dir, f'{osp.basename(data["A_paths"][b])}'))
