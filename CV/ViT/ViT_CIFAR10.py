"""
참고자료
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""
import os
import pathlib
import datetime as dt
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TVF

from ViT_v1 import ViT
from train_test_loop import train_loop, test_loop

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 0. Config
ROOT_PATH = "C:/Users/WHITE/Desktop/DATA/CIFAR10"  # C:/Users/BEGAS_15/Desktop/DATA/CIFAR10
img_size = (3, 32, 32)  # 이미지 크기
patch_size = 4  # 패치 크기
dim_model = 64  # 임베딩 벡터 차원
num_heads = 8  # MHA의 헤드 개수
n_enc_block = 6  # 인코더 블록 개수
batch_size = 128
MODEL_PATH = pathlib.Path("./models")

os.makedirs(MODEL_PATH, exist_ok=True)

start = dt.datetime.now().strftime("%Y%m%d_%H%M")

# 1. DATASETS & DATALOADERS
# 1.1. Loading a Dataset
train_dataset = datasets.CIFAR10(root=ROOT_PATH,
                                 train=True,
                                 download=True,
                                 transform=Compose([ToTensor(), Normalize((0, 0, 0), (1, 1, 1))])
                                 )
test_dataset = datasets.CIFAR10(root=ROOT_PATH,
                                train=False,
                                download=True,
                                transform=Compose([ToTensor(), Normalize((0, 0, 0), (1, 1, 1))])
                                )
n_class = len(train_dataset.classes)  # 클래스 개수

# [Optional] Iterating and Visualizing the Dataset
labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}
figure = plt.figure()
ncols, nrows = 5, 2
for i in range(1, ncols * nrows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    img = np.asarray(TVF.to_pil_image(img.detach()))

    figure.add_subplot(nrows, ncols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze())
plt.show()

# 1.2. Preparing your data for training with DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 2. Build the Neural Network
# 2.1. Get Device for Training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# 2.2. Define the Class
model = ViT(n_class=n_class, in_channels=img_size[0], img_size=img_size[1],
            patch_size=patch_size, dim_model=dim_model, num_heads=num_heads, n_enc_block=n_enc_block,
            mha_dropout=0.1, enc_dropout=0.1, ff_dropout=0.1, clsff_dropout=0.1,
            device=device
            )
# 2.3. Check the Number of Parameters
# summary(model, (3, 32, 32), device=device)

# 3. Optimizing Model Parameters
# 3.1. Hyperparameters
learning_rate = 1e-3
epochs = 50
# 3.2. Optimization Loop
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Adam은 왜 안 될까?

loss_hist = {"train": [], "val": []}
metric_hist = {"train": [], "val": []}
best_metric, best_epoch = 0, 0
print(f"Start Time: {start}\n----------------------------------------------------------------")
for epoch in range(epochs):
    epoch_start = dt.datetime.now()
    print(f"Epoch {epoch+1} / {epochs}")

    # Train
    train_loss, train_metric = train_loop(train_dataloader, model, loss_fn, optimizer, device=device, print_batch=False)
    loss_hist["train"].append(train_loss)
    metric_hist["train"].append(train_metric)
    print(f"Train Error: \n Accuracy: {(100 * train_metric):>0.1f}%, Avg loss: {train_loss:>7f}")

    # Validation
    test_loss, test_metric = test_loop(test_dataloader, model, loss_fn, device=device)
    loss_hist["val"].append(test_loss)
    metric_hist["val"].append(test_metric)
    print(f"Valid Error: \n Accuracy: {(100 * test_metric):>0.1f}%, Avg loss: {test_loss:>7f}")

    # Best
    if test_metric > best_metric:
        best_model = deepcopy(model)
        best_metric = deepcopy(test_metric)
        best_epoch = deepcopy(epoch)
        torch.save(best_model.state_dict(), MODEL_PATH / f"{start}_best_weights.pth")
        torch.save(best_model, MODEL_PATH / f"{start}_best_model.pth")
    epoch_end = dt.datetime.now()
    print(f"Best Accuracy: {(100*best_metric):>0.1f}% at Epoch {best_epoch + 1}, Epoch Lead Time: {(epoch_end-epoch_start).seconds/60:>0.1f} minutes."
          f"\n----------------------------------------------------------------")

print("Done!")
print(f"Start Time: {start} / End Time: {dt.datetime.now().strftime('%Y%m%d_%H%M')}"
      f"\n----------------------------------------------------------------")
