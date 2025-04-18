import torch.nn as nn
from lvit_models.deform import DeformableConv2d
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from lvit_models.MoE import MoE

model_urls = {
    'vgg19_no_BN': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}


def make_layers(cfg, batch_norm=True):
    '''
    :param cfg:
    :param batch_norm:
    :return:   construct vgg19 feature extraction layer.
    '''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'Baysian_Ma': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'Regular': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG_Baysian_Ma(nn.Module):
    '''
        class of the Model used in Baysian Loss.
    '''

    def __init__(self, features):
        super(VGG_Baysian_Ma, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            # DeformableConv2d(in_dim=512, out_dim=256, kernel_size=3, padding=1,  with_mask=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # DeformableConv2d(in_dim=256, out_dim=128, kernel_size=3, padding=1,  with_mask=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # DeformableConv2d(in_dim=128, out_dim=1, kernel_size=1, padding=0,  with_mask=False),
            nn.Conv2d(128, 1, 1)
        )
        self.recover = nn.Sequential()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, DeformableConv2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)

        return torch.abs(x)


def vgg19_d(use_batch_norm: bool = True, layers: str = 'Baysian_Ma', state_dict: str = None):
    """VGG 19-layer model (configuration "Baysian_Ma")
        model pre-trained on ImageNet
        the method return the model used for training.
    """
    model = VGG_Baysian_Ma(make_layers(cfg[layers], batch_norm=use_batch_norm))
    if state_dict is None:
        if use_batch_norm:
            pre_weight = torch.load('/home/ai1002/pretrained_official_weights/vgg19_bn-c79401a0.pth', map_location='cuda')
            miss, unexpect = model.load_state_dict(pre_weight, strict=False)
            z = 1
            # model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn'], 'Model/model_pretrain/checkpoints'),
            #                       strict=False)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19_no_BN'], 'Model/model_pretrain/checkpoints'),
                                  strict=False)
    else:
        model.load_state_dict(torch.load(state_dict))
    return model


if __name__ == '__main__':
    x = torch.rand(8, 3, 304, 304)
    model = vgg19_d()
    out = model(x)
    print(out.shape)
    z = 1
