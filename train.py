from datasets.unpaireddataset import UnpairedDataset
import torch
from torch.utils.data import DataLoader
from config import cfg
from models.cycle_gan import CycleGAN
from models.fcvgan import FCVGAN
from tqdm import tqdm
from common.utils.dir_utils import update_config
import argparse
import wandb


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
    parser.add_argument('--use_wandb',
                        help='use wandb',
                        action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    update_config(config_file=args.cfg, config=cfg)
    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project=cfg.project_name,
                   name=f'lr:{cfg.lr}_netG:{cfg.netG}_netD:{cfg.netD}')
        wandb.config.update({
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_thread,
            'optimizer': 'adam',
            'learning_rate': cfg.lr,
        })

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ********************
    # 1. Load datasets
    # ********************
    dataset = UnpairedDataset('train')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
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
    model = network(is_train=True)
    model.setup()

    # ****************
    # 3. Training
    # ****************
    for epoch in range(cfg.n_epochs + cfg.n_epochs_decay + 1):
        epoch_iter = 0
        for data in tqdm(dataloader):
            model.set_input(data)
            model.optimize_parameters()

            # log
            if use_wandb:
                tracknig = {
                    'loss_G_A': model.loss_G_A.cpu().detach(),
                    'loss_G_B': model.loss_G_B.cpu().detach(),
                    'loss_cycle_A': model.loss_cycle_A.cpu().detach(),
                    'loss_cycle_B': model.loss_cycle_B.cpu().detach(),
                    'loss_idt_A': model.loss_idt_A.cpu().detach(),
                    'loss_G': model.loss_G.cpu().detach(),
                    'loss_D_A': model.loss_D_A.cpu().detach(),
                    'loss_D_B': model.loss_D_B.cpu().detach()
                }
                wandb.log(tracknig)
        print('saving the model at the end of epoch %d' % (epoch))
        model.save_networks(epoch)
        model.update_learning_rate()
