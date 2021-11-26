from __future__ import absolute_import

import torch.nn as nn
import math

from ..activations import get_activation_module


def conv_bn(inp, oup, stride, activation: nn.Module):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True),
    )


def conv_1x1_bn(inp, oup, activation: nn.Module):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation: nn.Module):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self, n_class=1000, input_size=224, width_mult=1.0, activation="relu6"
    ):
        super(MobileNetV2, self).__init__()

        act = get_activation_module(activation)

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.layer0 = conv_bn(3, input_channel, 2, act)

        # building inverted residual blocks
        for layer_index, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)

            blocks = []
            for i in range(n):
                if i == 0:
                    blocks.append(
                        block(
                            input_channel,
                            output_channel,
                            s,
                            expand_ratio=t,
                            activation=act,
                        )
                    )
                else:
                    blocks.append(
                        block(
                            input_channel,
                            output_channel,
                            1,
                            expand_ratio=t,
                            activation=act,
                        )
                    )

                input_channel = output_channel

            self.add_module(f"layer{layer_index + 1}", nn.Sequential(*blocks))

        # building last several layers
        self.final_layer = conv_1x1_bn(input_channel, self.last_channel, activation=act)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.last_channel, n_class)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.final_layer(x)

        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def test():
    model = MobileNetV2().eval()
    print(model)
