import argparse
import datetime
import os
import pathlib

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Classifier
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_path", type=str, help="Path to folder containing samples"
    )
    parser.add_argument(
        "--classifier_path", type=str, default="classifier_ckpts/model.pt", help="Path to MNIST classifer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size'
    )
    parser.add_argument(
        "--label_of_dropped_class", type=int, default=0, help="Class label of forgotten class (for calculating average prob)"
    )

    args = parser.parse_args()
    return args


def parse_results_file(file_path):
    results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_experiment = None
    line_iter = iter(lines)  # イテレータを作成
    for line in line_iter:
        if 'nosa,ewc' in line or 'sa,ewc' in line:
            if current_experiment:
                results.append(current_experiment)
            current_experiment = {'type': line.strip(), 'data': []}
        elif 'forget:' in line:
            parts = line.split()
            forget = int(parts[1].strip(','))
            learn = int(parts[3])
            path = next(line_iter).strip()  # 次の行を読み取る
            entropy = float(next(line_iter).split(':')[1].strip())  # 次の行を読み取る
            prob = float(next(line_iter).split(':')[1].strip())  # 次の行を読み取る
            current_experiment['data'].append({'forget': forget, 'learn': learn, 'path': path, 'entropy': entropy, 'prob': prob})

    if current_experiment:
        results.append(current_experiment)

    return results




class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, n=None):
        self.transforms = transforms
        
        path = pathlib.Path(img_folder)
        self.files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        
        assert n is None or n <= len(self.files)
        self.n = len(self.files) if n is None else n
        
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('L')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def GetImageFolderLoader(path, batch_size):

    dataset = ImagePathDataset(
            path,
            transforms=transforms.ToTensor(),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )
    
    return loader

def evaluate_accuracy(model, data_loader, device, target_class):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += data.size(0)
            correct += (predicted == target_class).sum().item()
    
    return correct / total    
    
    
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    model = Classifier().to(device)
    model.eval()
    ckpt = torch.load(args.classifier_path, map_location=device)
    model.load_state_dict(ckpt)
    file_path = './2024_01_09_mnist_forget_learn_test.txt'  # テキストファイルのパスを指定
    experiments = parse_results_file(file_path)
    
    # 分類精度を保存する行列を作成
    # 10x10の行列を作成
    matrix_sa_ewc = [[0 for _ in range(10)] for _ in range(10)]
    matrix_nosa_ewc = [[0 for _ in range(10)] for _ in range(10)]
    
    for experiment in experiments:
        exp_type = experiment['type']
        for data in experiment['data']:
            forget = data['forget']
            learn = data['learn']
            folder_path = data['path']
            loader = GetImageFolderLoader(folder_path, args.batch_size)
            accuracy = evaluate_accuracy(model, loader, device, learn)
            print(f"Type: {exp_type}, Forget: {forget}, Learn: {learn}, Accuracy: {accuracy}")
            if exp_type == 'sa,ewc':
                matrix_sa_ewc[forget][learn] = accuracy
            elif exp_type == 'nosa,ewc':
                matrix_nosa_ewc[forget][learn] = accuracy
        
        # タイプごとに分類精度を保存
        
    # NumPy配列に変換
    matrix_sa_ewc_np = np.array(matrix_sa_ewc)
    matrix_nosa_ewc_np = np.array(matrix_nosa_ewc)


    # SA EWCとNoSA EWCの行列の差分を計算
    matrix_diff_np = matrix_sa_ewc_np - matrix_nosa_ewc_np
    max_abs_value = np.max(np.abs(matrix_diff_np))
    max_image_ratio = 1.0 + 0.1  # 図の最大値を調整する変数

    # 現在の日付と時間（分まで）を取得
    Time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

    # SA EWCの行列をプロット
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_sa_ewc_np, cmap='hot', interpolation='nearest')
    plt.title(f'SA EWC Classification Accuracy - {Time}')
    plt.colorbar()
    plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
    plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
    for i in range(10):
        for j in range(10):
            color = 'black' if matrix_sa_ewc_np[i][j] > 0.3 else 'white'
            plt.text(j, i, round(matrix_sa_ewc_np[i][j], 2), ha='center', va='center', color=color)
    plt.savefig(f'./matrix_sa_ewc_values_{Time}.png')
    plt.close()

    # NoSA EWCの行列をプロット
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_nosa_ewc_np, cmap='hot', interpolation='nearest')
    plt.title(f'NoSA EWC Classification Accuracy - {Time}')
    plt.colorbar()
    plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
    plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
    for i in range(10):
        for j in range(10):
            color = 'black' if matrix_nosa_ewc_np[i][j] > 0.3 else 'white'
            plt.text(j, i, round(matrix_nosa_ewc_np[i][j], 2), ha='center', va='center', color=color)
    plt.savefig(f'./matrix_nosa_ewc_values_{Time}.png')
    plt.close()

    # 差分行列をプロット
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_diff_np, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value*max_image_ratio, vmax=max_abs_value*max_image_ratio)
    plt.title(f'Difference between SA EWC and NoSA EWC - {Time}')
    plt.colorbar()
    plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
    plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
    for i in range(10):
        for j in range(10):
            color = 'black'
            plt.text(j, i, round(matrix_diff_np[i][j], 2), ha='center', va='center', color=color)
    plt.savefig(f'./matrix_diff_values_{Time}.png')
    plt.close()