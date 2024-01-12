import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import pickle
from model import OneHotCVAE, loss_function
from utils import setup_dirs
import os
import argparse
import logging
import copy
import numpy as np
from torchvision import datasets, transforms
from utils import get_config_and_setup_dirs, cycle



def parse_args_and_ckpt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ckpt_folder", type=str, required=True, help="Path to folder of original VAE"
    )
    
    parser.add_argument(
        "--data_path", type=str, default="./dataset", help="Path to MNIST dataset"
    )
    parser.add_argument(
        '--removed_label', type=int, default=0,help='an integer for no train label'
    )
    
 
    parser.add_argument(
        "--gamma", type=float, default = 1, help = "Gamma hyperparameter for contrastive term in loss (left at 1 in main paper)"
    )
    
    parser.add_argument(
        "--n_iters", type=int, default=10000, help="Number of iterations"
    )

    parser.add_argument(
        "--log_freq", type=int, default = 200, help='Logging frequency while training'
    )
    
    parser.add_argument(
        "--n_vis_samples", type=int, default=100, help='Number of samples to visualize while logging'
    )
    
    parser.add_argument(
        "--lr", type=float, default=1e-4, help='Learning rate'
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size for training'
    )
    
    args = parser.parse_args()
    ckpt = torch.load(os.path.join(args.ckpt_folder, "ckpts/ckpt.pt"), map_location=device)
    old_config = ckpt['config']
    new_config = setup_dirs(copy.deepcopy(old_config))
    
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(new_config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)
    
    return args, ckpt, old_config, new_config

def train():
    vae.train()
    
    train_loss = 0
    for step in range(0, args.n_iters):
        data, label = next(train_iter)
        label = F.one_hot(label, 10)
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data, label)
        loss = loss_function(recon_batch, data, mu, log_var)

        total_loss = loss
        
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()
        
        if (step+1) % args.log_freq == 0:
            logging.info('Train Step: {} ({:.0f}%)\t Avg Train Loss Per Batch: {:.6f}\t Avg Test Loss Per Batch: {:.6f}'.format(
                step, 100. * step / args.n_iters, train_loss / args.log_freq, test()))
            sample(step)
            train_loss = 0

            

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, label in test_loader:
            label = F.one_hot(label, 10)
            data = data.to(device)
            label = label.to(device)
            recon, mu, log_var = vae(data, label)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader)
    return test_loss

def sample(step):
    vae.eval()
    with torch.no_grad():
        z = torch.randn((args.n_vis_samples, new_config.z_dim)).to(device)
        c = torch.repeat_interleave(torch.arange(10), args.n_vis_samples//10).to(device)
        c = F.one_hot(c, 10)
        
        out = vae.decoder(z, c).view(-1, 1, 28, 28)
        
        grid = make_grid(out, nrow = args.n_vis_samples//10)
        save_image(grid, os.path.join(new_config.log_dir, "step_" + str(step) + ".png"))


def filter_labels(dataset, labels_to_keep):
    # データセットから指定されたラベルのみを保持
    mask = [label in labels_to_keep for label in dataset.targets]
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, ckpt, old_config, new_config = parse_args_and_ckpt()
    logging.info(f"Beginning EWC training of conditional VAE with new label ")
    
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transforms.ToTensor(), download=False)
    
    # fileter labels
    labels_to_keep = list(range(10))  
    # 忘れられたラベル
    removed_label = args.removed_label
    labels_to_keep.remove(removed_label)
    filter_labels(train_dataset, labels_to_keep)
    filter_labels(test_dataset, labels_to_keep)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    train_iter = cycle(train_loader)
    
    # build model
    vae = OneHotCVAE(x_dim=new_config.x_dim, h_dim1= new_config.h_dim1, h_dim2=new_config.h_dim2, z_dim=new_config.z_dim)
    vae = vae.to(device)

    vae.load_state_dict(ckpt['model'])
    vae.train()
    
    params_mle_dict = {}
    for name, param in vae.named_parameters():
        params_mle_dict[name] = param.data.clone()

       
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    
    train()
    torch.save({
            "model": vae.state_dict(),
            "config": new_config
        },
        os.path.join(new_config.ckpt_dir, "ckpt.pt"))
    print(f"finetuning save dir:{new_config.exp_root_dir}")