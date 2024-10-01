"""
Adaptation of Prof-of-Concept Network: StarNet.
Ma et al. makes StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Originally Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Adaptation by: Nick Eliopoulos (Email: neliopou@purdue.edu)
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from .layers.fusedlinear import FusedParallelLinear


###
### Original
### NOTE: bias=True implicitly for Conv2d
###
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


###
### Modified
###
class TritonStarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        ### Replaced!
        self.linear = FusedParallelLinear(dim, mlp_ratio)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x : torch.Tensor):
        y = self.dwconv(x)
        y = self.linear(y)
        y = self.dwconv2(self.g(y))
        y = x + self.drop_path(y)
        return y


class TritonStarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.SiLU())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [TritonStarBlock(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def triton_starnet_s1(pretrained=False, **kwargs):
    assert pretrained == False, "Unsupported"

    model = TritonStarNet(24, [2, 2, 8, 3], **kwargs)
    # if pretrained:
    #     url = model_urls['starnet_s1']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    #     model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def triton_starnet_s2(pretrained=False, **kwargs):
    assert pretrained == False, "Unsupported"

    model = TritonStarNet(32, [1, 2, 6, 2], **kwargs)
    # if pretrained:
    #     url = model_urls['starnet_s2']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    #     model.load_state_dict(checkpoint["state_dict"])
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = triton_starnet_s2()
    model.to(device)

    B = 32

    x = torch.randn(size=(B, 3, 224, 224), device=device, dtype=torch.float32)
    y = model(x)

    print(f"y shape: {y.shape}")