import argparse
import os
import pathlib
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 対応する画像ファイル拡張子のセット
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}



# ここには既存のコードを利用（ImagePathDataset, GetImageFolderLoader, Classifier）
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_path", type=str, help="root Path to folder containing samples" 
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "fashion"], help="Dataset to use (mnist or fashion)"
    )
    # parser.add_argument(
    #     "--classifier_path", type=str, default="classifier_ckpts/model.pt", help="Path to classifer"
    # )
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size'
    )
  

    args = parser.parse_args()
    return args


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

class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: ラベルごとのフォルダが含まれるルートディレクトリのパス
        transform: 画像に適用する変換（例：transforms.ToTensor()）
        """
        self.transform = transform
        self.samples = []

        for class_path in pathlib.Path(root_dir).iterdir():
            if class_path.is_dir() and class_path.name.split('_')[0].isdigit():
                label = int(class_path.name.split('_')[0])  # フォルダ名からラベルを取得
                for img_path in class_path.glob('*.*'):
                    if img_path.suffix[1:].lower() in IMAGE_EXTENSIONS:
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L') # 画像をグレースケールで読み込み
        if self.transform:
            image = self.transform(image)
        return image, label

def plot_confusion_matrix(cm, class_names):
    """
    混合行列を可視化する
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False, annot_kws={"size": 20})
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig('confusion_matrix_for_sensors.png')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.savefig('confusion_matrix_for_test.png')
    plt.show()

if __name__ == "__main__":
    args = parse_args()  # コマンドライン引数を解析
    path = args.sample_path  # サンプルのルートパス
    path = "./results/mnist/2024_03_12_164859_55"
    # 使用するGPUを指定
    gpu_index = 2  # 使用するGPUのインデックス
    torch.cuda.set_device(gpu_index)  # GPUを設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルをロード
    model = Classifier().to(device)
    model_load_path = f'./classifier_ckpts/model_{args.dataset}.pt'
    ckpt = torch.load(model_load_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # データローダを準備
    transform = transforms.Compose([transforms.ToTensor()])
    
    loader = DataLoader(ImagePathDataset(path, transform=transform), batch_size=args.batch_size, shuffle=False)

    true_labels = []
    pred_labels = []

    # データをモデルに渡して予測
    for images,labels in loader:
        # print(type(images))
        # images = images.to(device)
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        pred_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        # ここでは、実際のラベルが必要です。実際のラベルをどのように取得するかは、
        # データセットの実装に依存します。以下は仮のコードです。
        # true_labels.extend(ここに実際のラベルを追加)
    
    # 混合行列を計算
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 混合行列をプロット
    class_names = [str(i) for i in range(10)]  # クラス名を設定
    plot_confusion_matrix(cm, class_names)
