import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor
from utils.padding import ReflectionPad2d
from mindspore.common.initializer import initializer, Constant, One
import numpy as np
from mindspore.ops import functional as F
from utils.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
# from mindspore.ops.operations.nn_ops import AdaptiveMaxPool2D 
import pdb

class ResnetGenerator(nn.Cell):
    def __init__(self, ngf=64, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.light = light

        self.ConvBlock1 = nn.SequentialCell([ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=7, stride=1, pad_mode='pad', padding=0, has_bias=False),
                                       # nn.InstanceNorm2d(ngf),
                                       nn.BatchNorm2d(ngf),
                                       nn.ReLU()])

        self.HourGlass1 = HourGlass(ngf, ngf)
        self.HourGlass2 = HourGlass(ngf, ngf)

        # Down-Sampling
        self.DownBlock1 = nn.SequentialCell([ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, pad_mode='pad', padding=0, has_bias=False),
                                        # nn.InstanceNorm2d(ngf * 2),
                                        nn.BatchNorm2d(ngf * 2),
                                        nn.ReLU()])

        self.DownBlock2 = nn.SequentialCell([ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, pad_mode='pad', padding=0, has_bias=False),
                                        # nn.InstanceNorm2d(ngf*4),
                                        nn.BatchNorm2d(ngf*4),
                                        nn.ReLU()])

        # Encoder Bottleneck
        self.EncodeBlock1 = ResnetBlock(ngf*4)
        self.EncodeBlock2 = ResnetBlock(ngf*4)
        self.EncodeBlock3 = ResnetBlock(ngf*4)
        self.EncodeBlock4 = ResnetBlock(ngf*4)

        # Class Activation Map
        self.gap_fc = nn.Dense(in_channels=ngf * 4, out_channels=1)
        self.gmp_fc = nn.Dense(in_channels=ngf * 4, out_channels=1)
        self.conv1x1 = nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=1, stride=1, pad_mode='pad', has_bias=True)
        self.relu = nn.ReLU()

        # Gamma, Beta block
        if self.light:
            self.FC = nn.SequentialCell([nn.Dense(in_channels=ngf * 4, out_channels=ngf * 4),
                                    nn.ReLU(),
                                    nn.Dense(in_channels=ngf * 4, out_channels=ngf * 4),
                                    nn.ReLU()])
        else:
            self.FC = nn.SequentialCell([nn.Dense(in_channels=img_size // 4 * img_size // 4 * ngf * 4, out_channels=ngf * 4),
                                    nn.ReLU(),
                                    nn.Dense(in_channels=ngf * 4, out_channels=ngf * 4),
                                    nn.ReLU()])

        # Decoder Bottleneck
        self.DecodeBlock1 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock2 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock3 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock4 = ResnetSoftAdaLINBlock(ngf*4)

        # Up-Sampling
        self.rebilinear1 = nn.ResizeBilinear()
        self.UpBlock1 = nn.SequentialCell([ReflectionPad2d(1),
                                      nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=False),
                                      LIN(ngf*2),
                                      nn.ReLU()])

        self.rebilinear2 = nn.ResizeBilinear()
        self.UpBlock2 = nn.SequentialCell([ReflectionPad2d(1),
                                      nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=False),
                                      LIN(ngf),
                                      nn.ReLU()])

        self.HourGlass3 = HourGlass(ngf, ngf)
        self.HourGlass4 = HourGlass(ngf, ngf, False)

        self.ConvBlock2 = nn.SequentialCell([ReflectionPad2d(3),
                                        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, pad_mode='pad', padding=0, has_bias=False),
                                        nn.Tanh()])

        self.avgpool2d1 = AdaptiveAvgPool2d(output_size=(1))
        self.avgpool2d2 = AdaptiveAvgPool2d(output_size=(1))
        self.avgpool2d3 = AdaptiveAvgPool2d(output_size=(1))
        self.avgpool2d4 = AdaptiveAvgPool2d(output_size=(1))
        self.gap = AdaptiveAvgPool2d(output_size=(1))
        self.gmp = AdaptiveMaxPool2d(output_size=(1))
        self.avgpool2d_light = AdaptiveAvgPool2d(output_size=(1))

    def construct(self, x):
        x = self.ConvBlock1(x)
        x = self.HourGlass1(x)
        x = self.HourGlass2(x)

        x = self.DownBlock1(x)
        x = self.DownBlock2(x)

        x = self.EncodeBlock1(x)
        content_features1 = self.avgpool2d1(x)#.view(x.shape[0], -1,)
        content_features1 = content_features1.view(x.shape[0], -1)
        x = self.EncodeBlock2(x)
        content_features2 = self.avgpool2d2(x)#.view(x.shape[0], -1,)
        content_features2 = content_features2.view(x.shape[0], -1)
        x = self.EncodeBlock3(x)
        content_features3 = self.avgpool2d3(x)#.view(x.shape[0], -1,)
        content_features3 = content_features3.view(x.shape[0], -1)
        x = self.EncodeBlock4(x)
        content_features4 = self.avgpool2d4(x)#.view(x.shape[0], -1,)
        content_features4 = content_features4.view(x.shape[0], -1)

        gap = self.gap(x)
        gap = gap.view(x.shape[0], -1)
        gap_logit = self.gap_fc(gap)
        # gap_weight = list(self.gap_fc.get_parameters())[0]
        gap_weight = self.gap_fc.weight
        # gap_weight = gap_weight.unsqueeze(2).unsqueeze(3)
        gap_weight = ops.ExpandDims()(gap_weight, 2)
        gap_weight = ops.ExpandDims()(gap_weight, 3)
        gap = x * gap_weight#.unsqueeze(2).unsqueeze(3)

        gmp = self.gmp(x)
        gmp = gmp.view(x.shape[0], -1)
        gmp_logit = self.gmp_fc(gmp)
        # gmp_weight = list(self.gmp_fc.get_parameters())[0]
        gmp_weight = self.gmp_fc.weight 
        gmp_weight = ops.ExpandDims()(gmp_weight, 2)
        gmp_weight = ops.ExpandDims()(gmp_weight, 3)
        gmp = x * gmp_weight#.unsqueeze(2).unsqueeze(3)

        cam_logit = P.Concat(1)([gap_logit, gmp_logit])
        x = P.Concat(1)([gap, gmp])
        x = self.relu(self.conv1x1(x))

        heatmap = x.sum(axis=1, keepdims=True)

        if self.light:
            x_ = self.avgpool2d_light(x)
            style_features = self.FC(x_.view(x_.shape[0], -1))
        else:
            style_features = self.FC(x.view(x.shape[0], -1))

        x = self.DecodeBlock1(x, content_features4, style_features)
        x = self.DecodeBlock2(x, content_features3, style_features)
        x = self.DecodeBlock3(x, content_features2, style_features)
        x = self.DecodeBlock4(x, content_features1, style_features)

        x = self.rebilinear1(x, scale_factor=2)
        x = self.UpBlock1(x)
        x = self.rebilinear2(x, scale_factor=2)
        x = self.UpBlock2(x)

        x = self.HourGlass3(x)
        x = self.HourGlass4(x)
        out = self.ConvBlock2(x)

        return out, cam_logit, heatmap


class ConvBlock(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_out = dim_out

        self.ConvBlock1 = nn.SequentialCell([nn.BatchNorm2d(dim_in),
                                        # nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(),
                                        ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=dim_in, out_channels=dim_out // 2, kernel_size=3, stride=1, pad_mode='pad', has_bias=False)])

        self.ConvBlock2 = nn.SequentialCell([nn.BatchNorm2d(dim_out//2),
                                        # nn.InstanceNorm2d(dim_out//2),
                                        nn.ReLU(),
                                        ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=dim_out // 2, out_channels=dim_out // 4, kernel_size=3, stride=1, pad_mode='pad', has_bias=False)])

        self.ConvBlock3 = nn.SequentialCell([nn.BatchNorm2d(dim_out//4),
                                        # nn.InstanceNorm2d(dim_out//4),
                                        nn.ReLU(),
                                        ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=dim_out // 4, out_channels=dim_out // 4, kernel_size=3, stride=1, pad_mode='pad', has_bias=False)])

        self.ConvBlock4 = nn.SequentialCell([nn.BatchNorm2d(dim_in),
                                        # nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1, pad_mode='pad', has_bias=False)])

    def construct(self, x):
        residual = x

        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        out = P.Concat(1)((x1, x2, x3))

        if P.Shape()(residual)[1] != self.dim_out:
            residual = self.ConvBlock4(residual)

        return residual + out


class HourGlass(nn.Cell):
    def __init__(self, dim_in, dim_out, use_res=True):
        super(HourGlass, self).__init__()
        self.use_res = use_res

        self.HG = nn.SequentialCell([HourGlassBlock(dim_in, dim_out),
                                ConvBlock(dim_out, dim_out),
                                nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=1, stride=1, pad_mode='pad', has_bias=False),
                                nn.BatchNorm2d(dim_out),
                                # nn.InstanceNorm2d(dim_out),
                                nn.ReLU()])

        self.Conv1 = nn.Conv2d(in_channels=dim_out, out_channels=3, kernel_size=1, stride=1, pad_mode='pad', has_bias=True)

        if self.use_res:
            self.Conv2 = nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=1, stride=1, pad_mode='pad', has_bias=True)
            self.Conv3 = nn.Conv2d(in_channels=3, out_channels=dim_out, kernel_size=1, stride=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        ll = self.HG(x)
        tmp_out = self.Conv1(ll)

        if self.use_res:
            ll = self.Conv2(ll)
            tmp_out_ = self.Conv3(tmp_out)
            return x + ll + tmp_out_

        else:
            return tmp_out


class HourGlassBlock(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(HourGlassBlock, self).__init__()

        self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
        self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)

        self.ConvBlock5 = ConvBlock(dim_out, dim_out)

        self.ConvBlock6 = ConvBlock(dim_out, dim_out)
        self.ConvBlock7 = ConvBlock(dim_out, dim_out)
        self.ConvBlock8 = ConvBlock(dim_out, dim_out)
        self.ConvBlock9 = ConvBlock(dim_out, dim_out)

    def construct(self, x):
        skip1 = self.ConvBlock1_1(x)
        down1 = P.AvgPool(2, 2, 'valid')(x)
        down1 = self.ConvBlock1_2(down1)

        skip2 = self.ConvBlock2_1(down1)
        down2 = P.AvgPool(2, 2, 'valid')(down1)
        down2 = self.ConvBlock2_2(down2)

        skip3 = self.ConvBlock3_1(down2)
        down3 = P.AvgPool(2, 2, 'valid')(down2)
        down3 = self.ConvBlock3_2(down3)

        skip4 = self.ConvBlock4_1(down3)
        down4 = P.AvgPool(2, 2, 'valid')(down3)
        down4 = self.ConvBlock4_2(down4)

        center = self.ConvBlock5(down4)

        up4 = self.ConvBlock6(center)
        up4 = nn.ResizeBilinear()(up4, scale_factor=2)
        up4 = skip4 + up4

        up3 = self.ConvBlock7(up4)
        up3 = nn.ResizeBilinear()(up3, scale_factor=2)
        up3 = skip3 + up3

        up2 = self.ConvBlock8(up3)
        up2 = nn.ResizeBilinear()(up2, scale_factor=2)
        up2 = skip2 + up2

        up1 = self.ConvBlock9(up2)
        up1 = nn.ResizeBilinear()(up1, scale_factor=2)
        up1 = skip1 + up1

        return up1


class ResnetBlock(nn.Cell):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias),
                       nn.BatchNorm2d(dim),
                       # nn.InstanceNorm2d(dim),
                       nn.ReLU()]

        conv_block += [ReflectionPad2d(1),
                       nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias),
                       # nn.InstanceNorm2d(dim),
                       nn.BatchNorm2d(dim)]

        self.conv_block = nn.SequentialCell([*conv_block])

    def construct(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetSoftAdaLINBlock(nn.Cell):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def construct(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x


class ResnetAdaLINBlock(nn.Cell):
    def __init__(self, dim, use_bias=False):
        super(ResnetAdaLINBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias)
        self.norm1 = adaLIN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=0, has_bias=use_bias)
        self.norm2 = adaLIN(dim)

    def construct(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class SoftAdaLIN(nn.Cell):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)

        self.w_gamma = Parameter(ms.numpy.zeros((1, num_features)))
        self.w_beta = Parameter(ms.numpy.zeros((1, num_features)))

        self.c_gamma = nn.SequentialCell([nn.Dense(in_channels=num_features, out_channels=num_features),
                                     nn.ReLU(),
                                     nn.Dense(in_channels=num_features, out_channels=num_features)])
        self.c_beta = nn.SequentialCell([nn.Dense(in_channels=num_features, out_channels=num_features),
                                    nn.ReLU(),
                                    nn.Dense(in_channels=num_features, out_channels=num_features)])
        self.s_gamma = nn.Dense(in_channels=num_features, out_channels=num_features)
        self.s_beta = nn.Dense(in_channels=num_features, out_channels=num_features)

    def construct(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)

        # w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        w_gamma = ops.BroadcastTo((x.shape[0], -1))(self.w_gamma)
        w_beta = ops.BroadcastTo((x.shape[0], -1))(self.w_beta)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out


class adaLIN(nn.Cell):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        # self.weight = Parameter(Tensor(np.ones((1, 2)), ms.float32), name="w", requires_grad=True)

        # x = initializer(Constant(0.9), [1, num_features, 1, 1], ms.float32)
        self.rho = Parameter(Tensor(shape = (1, num_features, 1, 1), init=Constant(0.9), dtype=ms.float32))
        # self.rho.data.fill_(0.9)

    def construct(self, input, gamma, beta):
        in_mean, in_var = input.mean(axis=(2, 3), keep_dims=True), input.var(axis=(2, 3), keepdims=True)
        out_in = (input - in_mean) / P.Sqrt()(in_var + self.eps)
        ln_mean, ln_var = input.mean(axis=(1, 2, 3), keep_dims=True), input.var(axis=(1, 2, 3), keepdims=True)
        out_ln = (input - ln_mean) / P.Sqrt()(ln_var + self.eps)
        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        rho = ops.BroadcastTo((input.shape[0], -1, -1, -1))(self.rho)
        out = rho * out_in + (1-rho) * out_ln
        # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        gamma = ops.ExpandDims()(gamma, 2)
        gamma = ops.ExpandDims()(gamma, 3)
        beta = ops.ExpandDims()(beta, 2)
        beta = ops.ExpandDims()(beta, 3)
        out = out * gamma + beta

        return out


class LIN(nn.Cell):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(Tensor(shape=(1, num_features, 1, 1), init=Constant(0.0), dtype=ms.float32))
        self.gamma = Parameter(Tensor(shape=(1, num_features, 1, 1), init=Constant(1.0), dtype=ms.float32))
        self.beta = Parameter(Tensor(shape=(1, num_features, 1, 1), init=Constant(0.0), dtype=ms.float32))
        # self.rho.data.fill_(0.0)
        # self.gamma.data.fill_(1.0)
        # self.beta.data.fill_(0.0)

    def construct(self, input):
        in_mean, in_var = input.mean(axis=[2, 3], keep_dims=True), input.var(axis=(2, 3), keepdims=True)
        out_in = (input - in_mean) / P.Sqrt()(in_var + self.eps)
        ln_mean, ln_var = input.mean(axis=[1, 2, 3], keep_dims=True), input.var(axis=(1, 2, 3), keepdims=True)
        out_ln = (input - ln_mean) / P.Sqrt()(ln_var + self.eps)
        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        broadcast = ops.BroadcastTo((input.shape[0], -1, -1, -1))
        rho = broadcast(self.rho)
        out = rho * out_in + (1-rho) * out_ln
        # out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        gamma, beta = broadcast(self.gamma), broadcast(self.beta)
        out = out * gamma + beta

        return out


class Discriminator(nn.Cell):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(1),
                 nn.Conv2d(in_channels=input_nc, out_channels=ndf, kernel_size=4, stride=2, pad_mode='pad', padding=0, has_bias=True),
                 nn.LeakyReLU(alpha=0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      nn.Conv2d(in_channels=ndf * mult, out_channels=ndf * mult * 2, kernel_size=4, stride=2, pad_mode='pad', padding=0, has_bias=True),
                      nn.LeakyReLU(alpha=0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ndf * mult, out_channels=ndf * mult * 2, kernel_size=4, stride=1, pad_mode='pad', padding=0, has_bias=True),
                  nn.LeakyReLU(alpha=0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.Dense(in_channels=ndf * mult, out_channels=1, has_bias=False)
        self.gmp_fc = nn.Dense(in_channels=ndf * mult, out_channels=1, has_bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=ndf * mult * 2, out_channels=ndf * mult, kernel_size=1, stride=1, pad_mode='pad', has_bias=True)
        self.leaky_relu = nn.LeakyReLU(alpha=0.2)

        self.pad = ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels=ndf * mult, out_channels=1, kernel_size=4, stride=1, pad_mode='pad', padding=0, has_bias=False)

        self.model = nn.SequentialCell([*model])

        self.gap_avgpool2d = AdaptiveAvgPool2d(output_size=(1))
        self.gmp_avgpool2d = AdaptiveAvgPool2d(output_size=(1))

    def construct(self, input):
        x = self.model(input)

        gap = self.gap_avgpool2d(x)
        # gap_logit = self.gap_fc(gap, (x.shape[0], -1,))
        gap = gap.view(x.shape[0], -1)
        gap_logit = self.gap_fc(gap)
        # gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = self.gap_fc.weight
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gap_weight = ops.ExpandDims()(gap_weight, 2)
        gap_weight = ops.ExpandDims()(gap_weight, 3)
        gap = x * gap_weight

        gmp = self.gmp_avgpool2d(x)
        gmp = gmp.view(x.shape[0], -1)
        gmp_logit = self.gmp_fc(gmp)
        # gmp = mindspore.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp, (x.shape[0], -1,))
        gmp_weight = self.gmp_fc.weight 
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp_weight = ops.ExpandDims()(gmp_weight, 2)
        gmp_weight = ops.ExpandDims()(gmp_weight, 3)
        gmp = x * gmp_weight

        cam_logit = P.Concat(1)([gap_logit, gmp_logit])
        x = P.Concat(1)([gap, gmp])
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = x.sum(axis=1, keepdims=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class WClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'w_gamma'):
            w = module.w_gamma.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_gamma.data = w

        if hasattr(module, 'w_beta'):
            w = module.w_beta.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.w_beta.data = w
