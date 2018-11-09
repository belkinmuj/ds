from torch import nn
from torch.nn import functional as F
import torch
#from torchvision import models

from senet import se_resnet152
from collections import OrderedDict

def load_from_path(checkpoint_path, model):
    state = torch.load(checkpoint_path)
#    state = state["weight"]

#    new_state_dict = OrderedDict()
#    for k, v in state.items():
#        name = k[7:] # remove `module.`
#        new_state_dict[name] = v
#    model.load_state_dict(new_state_dict)
    model.load_state_dict(state["state_dict"])
    #print('model loaded from %s' % checkpoint_path)


def load_model(pretrained_path=None):

    model = se_resnet152()

#    load_from_path(pretrained_path, model)

    return model


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x



class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Resnet152unet(nn.Module):
    """
        """

    def __init__(self, num_classes=1, num_filters=32, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = load_model()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.size())
        conv2 = self.conv2(conv1)
        # print(conv2.size())
        conv3 = self.conv3(conv2)
        # print(conv3.size())
        conv4 = self.conv4(conv3)
        # print(conv4.size())
        conv5 = self.conv5(conv4)
        # print(conv5.size())

        center = self.center(self.pool(conv5))

        # print(center.size(), conv5.size())

        #print(self.center.in_channels, self.center.middle_channels, self.center.out_channels)

        dec5 = self.dec5(torch.cat([center, conv5], 1))


        # print("DEC5",dec5.size())

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))

        # print(dec4.size())
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # print(dec3.size())
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # print(dec2.size())
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)


        x_out = self.final(dec0)

        return x_out

# model = Resnet152unet().cuda()
# inp = torch.rand((2, 3, 128, 128)).cuda()
# out = model(inp)
# print(out.size())
