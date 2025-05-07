# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.resnet import ResNet, make_blocks, create_classifier
from collections import OrderedDict

class SaveOutput:
    # stores the hooked layers
    # Source 1: https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca
    # Source 2: https://discuss.pytorch.org/t/extracting-stats-from-activated-layers-training/45880/2
    def __init__(self):
        self.outputs = OrderedDict()

    def save_activation(self, name):
        def hook(module, module_in, module_out):
            self.outputs.update({name: module_out})

        return hook

# hack: inject the `get_downsample_ratio` function into `timm.models.resnet.ResNet`
def get_downsample_ratio(self: ResNet) -> int:
    return 32


# hack: inject the `get_feature_map_channels` function into `timm.models.resnet.ResNet`
def get_feature_map_channels(self: ResNet) -> List[int]:
    # `self.feature_info` is maintained by `timm`
    return [info['num_chs'] for info in self.feature_info[1:]]


# hack: override the forward function of `timm.models.resnet.ResNet`
def forward(self, x, hierarchical=False):
    """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
    >>> ResNet.forward
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    
    if hierarchical:
        ls = []
        x = self.layer1(x); ls.append(x)
        x = self.layer2(x); ls.append(x)
        x = self.layer3(x); ls.append(x)
        x = self.layer4(x); ls.append(x)
        self.global_pool(x)
        return ls
    else:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x

# hack: override the __init__ function of `timm.models.resnet.ResNet`
def init(self, block, layers, num_classes=1000, in_chans=3,
         cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False,
         output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False,
         act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.,
         drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
    block_args = block_args or dict()
    assert output_stride in (8, 16, 32)
    self.num_classes = num_classes
    self.drop_rate = drop_rate
    super(ResNet, self).__init__()

    # Stem
    deep_stem = 'deep' in stem_type
    inplanes = stem_width * 2 if deep_stem else 64
    if deep_stem:
        stem_chs = (stem_width, stem_width)
        if 'tiered' in stem_type:
            stem_chs = (3 * (stem_width // 4), stem_width)
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
            norm_layer(stem_chs[0]),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
            norm_layer(stem_chs[1]),
            act_layer(inplace=True),
            nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
    else:
        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(inplanes)
    self.act1 = act_layer(inplace=True)
    self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

    # Stem Pooling
    if replace_stem_pool:
        self.maxpool = nn.Sequential(*filter(None, [
            nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
            aa_layer(channels=inplanes, stride=2) if aa_layer else None,
            norm_layer(inplanes),
            act_layer(inplace=True)
        ]))
    else:
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Feature Blocks
    channels = [64, 128, 256, 512]
    stage_modules, stage_feature_info = make_blocks(
        block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
        output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
        down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
        drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
    for stage in stage_modules:
        self.add_module(*stage)  # layer1, layer2, etc
    self.feature_info.extend(stage_feature_info)

    # Head (Pooling and Classifier)
    self.num_features = 512 * block.expansion
    self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    self.init_weights(zero_init_last_bn=zero_init_last_bn)

    # feature extraction
    self.save_output = SaveOutput()
    self.hook_map = OrderedDict([
        ("B1U1", ["layer1", 0]),
        ("B1U2", ["layer1", 1]),
        ("B1U3", ["layer1", 2]),
        ("B2U1", ["layer2", 0]),
        ("B2U2", ["layer2", 1]),
        ("B2U3", ["layer2", 2]),
        ("B2U4", ["layer2", 3]),
        ("B3U1", ["layer3", 0]),
        ("B3U2", ["layer3", 1]),
        ("B3U3", ["layer3", 2]),
        ("B3U4", ["layer3", 3]),
        ("B3U5", ["layer3", 4]),
        ("B3U6", ["layer3", 5]),
        ("B4U1", ["layer4", 0]),
        ("B4U2", ["layer4", 1]),
        ("TCL", ["layer4", 2]),
        ("POOL", ["global_pool"]),
    ])
    for region in self.hook_map:
        hook_loc = self.hook_map[region]
        layer = getattr(self, hook_loc[0])
        if len(hook_loc) == 2:
            layer = layer[hook_loc[1]]
        layer.register_forward_hook(self.save_output.save_activation(region))

ResNet.get_downsample_ratio = get_downsample_ratio
ResNet.get_feature_map_channels = get_feature_map_channels
ResNet.forward = forward
ResNet.__init__ = init


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet50')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch

    # check feature extraction
    print(cnn.save_output.outputs)


if __name__ == '__main__':
    convnet_test()
