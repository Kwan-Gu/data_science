import torchvision
from torch import nn


train_dataset = torchvision.datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                             train=True,
                                             download=True
                                             )
test_dataset = torchvision.datasets.CIFAR10(root="C:/Users/BEGAS_15/Desktop/DATA/CIFAR10",
                                            train=False,
                                            download=True
                                            )

n_class = 10  # 클래스 개수
img_size = (32, 32, 3)  # 이미지 크기

patch_size = 4  # 패치 크기
p2 = (img_size[0]//patch_size)**2  # 패치 개수
d_model = 64  # 임베딩 벡터 차원
h = 8  # MHA의 헤드 개수
N = 6  # 인코더 블록 개수

class Patches(nn.Module):
    def __init__(self):