import os
import os.path as osp
from abc import ABC

import cv2
import torch
from common.utils.imutils import load_rgb_img
from datasets.dataset import BaseDataset
import constants
from glob import glob
from config import cfg
import random
random.seed(0)


class UnpairedDataset(BaseDataset, ABC):
    def __init__(self, data_split):
        super(UnpairedDataset, self).__init__(data_split)
        A_dir = osp.join(constants.ROOT_DATA, f'{data_split}A')
        B_dir = osp.join(constants.ROOT_DATA, f'{data_split}B')
        self.A_paths, self.B_paths, \
            self.A_size, self.B_size = self.load_data(A_dir, B_dir)
        self.transform = cfg.data_transforms[data_split]

    def load_data(self, A_dir, B_dir):
        types = ['*.png', '*.jpg']
        A_paths, B_paths = [], []
        for _type in types:
            A_paths.extend(glob(f'{A_dir}/{_type}'))
            B_paths.extend(glob(f'{B_dir}/{_type}'))
        A_size, B_size = len(A_paths), len(B_paths)
        print(f'End loading data. The number of data: A size:{A_size}, B size:{B_size}')
        return A_paths, B_paths, A_size, B_size

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = load_rgb_img(A_path)
        B_img = load_rgb_img(B_path)

        # apply transform
        A = self.transform(image=A_img)['image']
        B = self.transform(image=B_img)['image']

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
        }

    def __len__(self):
        return max(self.A_size, self.B_size)


if __name__ == '__main__':
    dataset = UnpairedDataset('test')
    for i, item in enumerate(dataset):
        A = item['A']
        B = item['B']
        A_paths = item['A_paths']
        B_paths = item['B_paths']

        # vis
        cv2.imwrite(f'{cfg.vis_dir}/sample_{osp.basename(A_paths)}', A[...,::-1])
        cv2.imwrite(f'{cfg.vis_dir}/sample_{osp.basename(B_paths)}', B[...,::-1])
        break
