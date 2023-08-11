"""
참고자료
- 컴퓨터 비전과 딥러닝 (오일석 저)
- https://github.com/FrancescoSaverioZuppichini/ViT
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchsummary import summary


def extract_patches(img: Tensor, patch_size: int, stride: int) -> Tensor:
    patches = (img.unfold(2, patch_size, stride)
               .unfold(3, patch_size, stride))  # (B, C, H/patch_size, W/patch_size, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patch, num_patch, C, patch_size, patch_size)
    patches = patches.reshape(*patches.size()[: 3], -1)  # (B, num_patch, num_patch, C*patch_size*patch_size)
    return patches


class Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, img: Tensor) -> Tensor:
        batch_size = img.shape[0]
        patches = extract_patches(img, patch_size=self.patch_size, stride=self.patch_size)
        patch_dims = patches.shape[-1]  # C*patch_size*patch_size
        patches = torch.reshape(patches, (batch_size, -1, patch_dims))  # (B, num_patch2, C*patch_size*patch_size)
        return patches


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, num_patch2: int, dim_model: int, device="cpu"):
        super().__init__()
        self.num_patch2 = num_patch2
        self.device = device
        self.projection = nn.Linear(in_features=in_channels*patch_size*patch_size, out_features=dim_model,
                                    device=device
                                    )
        self.position_embedding = nn.Embedding(num_patch2, dim_model, device=device)

    def forward(self, patch: Tensor) -> Tensor:
        positions = torch.unsqueeze(torch.arange(0, self.num_patch2, step=1), 0).to(self.device)
        embedded = torch.add(self.projection(patch.to(self.device)), self.position_embedding(positions))  # (B, num_patch2, dim_model)
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
        attention = F.softmax(qkt, dim=-1)/((self.dim_model // self.num_heads) ** 0.5)
        attention = self.drop(attention)
        context = torch.einsum("bhan, bhnd -> bhad", attention, vs)  # (B, num_heads, num_patch2, dim_model/num_heads)
        context = context.permute(0, 2, 1, 3)  # (B, num_patch2, num_heads, dim_model/num_heads)
        context = context.reshape(*context.shape[: 2], -1)  # (B, num_patch2, dim_model)
        context = self.projection(context)  # (B, num_patch2, dim_model)
        return context


class ResidualAdd(nn.Module):
    def __init__(self, func, device="cpu"):
        super().__init__()
        self.func = func.to(device)
        self.device = device

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        residual = x.to(self.device)

        x = self.func(x, **kwargs).to(self.device)
        x += residual.to(self.device)
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
    def __init__(self, dim_model: int, num_heads: int,
                 enc_dropout: float = 0, ff_dropout: float = 0, ff_expansion: int = 2, device="cpu",
                 **kwargs
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_model, eps=1e-6),
                MultiHeadAttention(dim_model, num_heads, **kwargs),
                nn.Dropout(enc_dropout)
            ), device=device),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_model, eps=1e-6),
                FeedFoward(dim_model, ff_expansion, ff_dropout),
                nn.Dropout(enc_dropout)
            ), device=device)
        )


class ClassificationFeedForward(nn.Sequential):
    def __init__(self, dim_model: int, num_patch2: int, n_class: int,
                 clsff_dropout: float = 0, dim1: int = 2048, dim2: int = 1024, device="cpu"
                 ):
        super().__init__(
            nn.LayerNorm(dim_model, eps=1e-6, device=device),  # (B, num_patch2, dim_model)
            nn.Flatten(),  # (B, num_patch2*dim_model)
            nn.Dropout(clsff_dropout),
            nn.Linear(num_patch2 * dim_model, dim1, device=device),  # (B, dim1)
            nn.GELU(),
            nn.Dropout(clsff_dropout),
            nn.Linear(dim1, dim2, device=device),  # (B, dim2)
            nn.GELU(),
            nn.Dropout(clsff_dropout),
            nn.Linear(dim2, n_class, device=device)  # (B, n_class)
        )


class ViT(nn.Module):
    def __init__(self, n_class: int, in_channels: int, img_size: int, patch_size: int,
                 dim_model: int, num_heads: int, n_enc_block: int, device="cpu",
                 mha_dropout=0.1, enc_dropout=0.1, ff_dropout=0.1, clsff_dropout=0.1,
                 **kwargs
                 ):
        super().__init__()
        num_patch2 = (img_size//patch_size)**2
        self.n_class = n_class
        self.patches = Patches(patch_size)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, num_patch2, dim_model, device=device)
        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(dim_model, num_heads, mha_dropout=mha_dropout, enc_dropout=enc_dropout, ff_dropout=ff_dropout,
                         device=device, **kwargs
                         )
            for _ in range(n_enc_block)
        ])
        self.classification = ClassificationFeedForward(dim_model, num_patch2, n_class, clsff_dropout=clsff_dropout,
                                                        device=device
                                                        )

    def forward(self, inputs: Tensor) -> Tensor:
        patches = self.patches(inputs)  # (B, num_patch2, C*patch_size*patch_size)
        embedded_patches = self.patch_embedding(patches)  # (B, num_patch2, dim_model)
        x = self.encoder_blocks(embedded_patches)  # (B, num_patch2, dim_model)
        x = self.classification(x)  # (B, n_class)
        output = nn.Softmax(dim=-1)(x)
        return output


if __name__ == "__main__":
    summary(ViT(n_class=10, in_channels=3, img_size=32, patch_size=4, dim_model=64, num_heads=8, n_enc_block=6),
            (3, 32, 32),
            device="cpu"
            )
