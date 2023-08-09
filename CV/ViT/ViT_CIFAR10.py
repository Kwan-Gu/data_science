import torch
import torchvision
from torch import Tensor, nn

train_dataset = torchvision.datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                             train=True,
                                             download=True
                                             )
test_dataset = torchvision.datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                            train=False,
                                            download=True
                                            )

n_class = 10  # 클래스 개수
img_size = (3, 32, 32)  # 이미지 크기

patch_size = 4  # 패치 크기
num_patch2 = (img_size[1]//patch_size) ** 2  # 패치 개수
dim_model = 64  # 임베딩 벡터 차원
h = 8  # MHA의 헤드 개수
N = 6  # 인코더 블록 개수


def extract_patches(img, n_patch_size, stride):
    patches = (img.unfold(2, n_patch_size, stride)
               .unfold(3, n_patch_size, stride))  # (B, C, H/patch_size, W/patch_size, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patch, num_patch, C, patch_size, patch_size)
    patches = patches.view(*patches.size()[: 3], -1)  # (B, num_patch, num_patch, C*patch_size*patch_size)
    return patches


class Patches(nn.Module):
    def __init__(self, n_patch_size):
        super(Patches, self).__init__()
        self.patch_size = n_patch_size

    def forward(self, img):
        batch_size = torch.Size(img)[0]
        patches = extract_patches(img, n_patch_size=self.patch_size, stride=self.patch_size)
        patch_dims = patches.shape[-1]  # C*patch_size*patch_size
        patches = torch.reshape(patches, (batch_size, -1, patch_dims))  # (B, num_patch2, C*patch_size*patch_size)
        return patches


class PatchEncoder(nn.Module):
    def __init__(self, num_patch2, dim_model):
        super(PatchEncoder, self).__init__()
        self.num_patch2 = num_patch2
        self.projection = nn.Linear(in_features=img_size[0]*patch_size*patch_size, out_features=dim_model)
        self.position_embedding = nn.Embedding(num_patch2, dim_model)

    def forward(self, patch):
        positions = torch.range(0, self.num_patch2, step=1)
        encoded = self.projection(patch)+self.position_embedding(positions)  # (dim_model)
        return encoded

def create_vit_classifier():
    input =