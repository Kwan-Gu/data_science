"""
참고자료
- https://github.com/FrancescoSaverioZuppichini/ViT
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- 컴퓨터 비전과 딥러닝 (오일석 저)
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TVF

import numpy as np

import matplotlib.pyplot as plt

# 0. Config
img_size = (3, 32, 32)  # 이미지 크기
patch_size = 4  # 패치 크기
num_patch2 = (img_size[-1] // patch_size) ** 2  # 패치 개수
dim_model = 64  # 임베딩 벡터 차원
num_heads = 8  # MHA의 헤드 개수
n_enc_block = 6  # 인코더 블록 개수

# 1. DATASETS & DATALOADERS
# 1.1. Loading a Dataset
train_dataset = datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                 train=True,
                                 download=True,
                                 transform=Compose([ToTensor(), Normalize((0, 0, 0), (1, 1, 1))])
                                 )
test_dataset = datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                train=False,
                                download=True,
                                transform=Compose([ToTensor(), Normalize((0, 0, 0), (1, 1, 1))])
                                )
n_class = len(train_dataset)  # 클래스 개수
"""
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
"""
# 1.2. Preparing your data for training with DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

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
def extract_patches(img, patch_size, stride):
    patches = (img.unfold(2, patch_size, stride)
               .unfold(3, patch_size, stride))  # (B, C, H/patch_size, W/patch_size, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patch, num_patch, C, patch_size, patch_size)
    patches = patches.reshape(*patches.size()[: 3], -1)  # (B, num_patch, num_patch, C*patch_size*patch_size)
    return patches


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img: Tensor) -> Tensor:
        batch_size = img.shape[0]
        patches = extract_patches(img, patch_size=self.patch_size, stride=self.patch_size)
        patch_dims = patches.shape[-1]  # C*patch_size*patch_size
        patches = torch.reshape(patches, (batch_size, -1, patch_dims))  # (B, num_patch2, C*patch_size*patch_size)
        return patches


class PatchEmbedding(nn.Module):
    def __init__(self, num_patch2, dim_model):
        super().__init__()
        self.num_patch2 = num_patch2
        self.projection = nn.Linear(in_features=img_size[0] * patch_size * patch_size, out_features=dim_model)
        self.position_embedding = nn.Embedding(num_patch2, dim_model)

    def forward(self, patch):
        positions = torch.unsqueeze(torch.arange(0, self.num_patch2, step=1), 0)
        embedded = self.projection(patch) + self.position_embedding(positions)  # (B, num_patch2, dim_model)
        return embedded


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, mha_dropout: float = 0.):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.queries = nn.Linear(dim_model, dim_model)  # dim_model = dim_key * num_heads
        self.keys = nn.Linear(dim_model, dim_model)  # dim_model = dim_key * num_heads
        self.values = nn.Linear(dim_model, dim_model)  # dim_model = dim_value * num_heads
        self.drop = nn.Dropout(mha_dropout)
        self.projection = nn.Linear(dim_model, dim_model)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qs = self.queries(x)  # (B, num_patch2, dim_model)
        ks = self.keys(x)  # (B, num_patch2, dim_model)
        vs = self.values(x)  # (B, num_patch2, dim_model)

        qs = qs.reshape(*qs.shape[:-1], self.num_heads, -1)  # (B, num_patch2, num_heads, dim_model/num_heads
        ks = ks.reshape(*ks.shape[:-1], self.num_heads, -1)  # (B, num_patch2, num_heads, dim_model/num_heads
        vs = vs.reshape(*vs.shape[:-1], self.num_heads, -1)  # (B, num_patch2, num_heads, dim_model/num_heads

        qs = qs.permute(0, 2, 1, 3)  # (B, num_heads, num_patch2, dim_model/num_heads
        ks = ks.permute(0, 2, 1, 3)  # (B, num_heads, num_patch2, dim_model/num_heads
        vs = vs.permute(0, 2, 1, 3)  # (B, num_heads, num_patch2, dim_model/num_heads

        qkt = torch.einsum("bhqd, bhkd -> bhqk", qs, ks)  # (B, num_heads, num_patch2, num_patch2)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            qkt.mask_fill(~mask, fill_value)
        attention = F.softmax(qkt / ((self.dim_model // self.num_heads) ** 0.5), dim=-1)
        attention = self.drop(attention)
        context = torch.einsum("bhan, bhnd -> bhad", attention, vs)  # (B, num_heads, num_patch2, dim_model/num_heads)
        context = context.permute(0, 2, 1, 3)  # (B, num_patch2, num_heads, dim_model/num_heads)
        context = context.reshape(*context.shape[: 2], -1)  # (B, num_patch2, dim_model)
        context = self.projection(context)  # (B, num_patch2, dim_model)
        return context


class ResidualAdd(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, **kwargs):
        residual = x

        x = self.func(x, **kwargs)
        x += residual
        return x


class FeedFoward(nn.Sequential):
    def __init__(self, dim_model: int, ff_expansion: int = 2, ff_dropout: float = 0.):
        super().__init__(
            nn.Linear(dim_model, dim_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim_model * ff_expansion, dim_model)
        )


class EncoderBlock(nn.Sequential):
    def __init__(self, dim_model: int, num_heads: int, enc_dropout: float = 0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_model, eps=1e-6),
                MultiHeadAttention(dim_model, num_heads, **kwargs),
                nn.Dropout(enc_dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_model, eps=1e-6),
                FeedFoward(dim_model, **kwargs),
                nn.Dropout(enc_dropout)
            ))
        )


class ClassificationFeedForward(nn.Sequential):
    def __init__(self, dim_model, n_class, clsff_dropout: float = 0, dim1: int = 2048, dim2: int = 1024):
        super().__init__(
            nn.LayerNorm(dim_model, eps=1e-6),
            nn.Flatten(),
            nn.Dropout(clsff_dropout),
            nn.Linear(dim_model, dim1),
            nn.GELU(),
            nn.Dropout(clsff_dropout),
            nn.Linear(dim1, dim2),
            nn.GELU(),
            nn.Dropout(clsff_dropout),
            nn.Linear(dim2, n_class)
        )


class ViT(nn.Module):
    def __init__(self, patch_size, num_patch2, dim_model, n_enc_block):
        super().__init__()
        self.patches = Patches(patch_size)
        self.patch_embedding = PatchEmbedding(num_patch2, dim_model)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(dim_model, num_heads, **kwargs) for _ in range(n_enc_block)])
        self.classification = ClassificationFeedForward(dim_model, n_class)

    def forward(self, inputs):
        patches = self.patches(inputs)  # (B, num_patch2, C*patch_size*patch_size)
        embedded_patches = self.patch_embedding(patches)  # (B, num_patch2, dim_model)
        x = self.encoder_blocks(embedded_patches)  # (B, num_patch2, dim_model)
        x = self.classification(x)  # (B, n_class)
        output = nn.Softmax()(x)
        return output

