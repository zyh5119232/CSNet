import torch
import torch.nn as nn
import torch.nn.functional as F
from config_convnext import arg
from backbone.convnext import convnext_base
from backbone.ResNet import Backbone_ResNet50_in3
from backbone.VGG import Backbone_vgg16_in3

class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size=(3,3), bias=True, stride=1, padding=(1,1), groups=1, reduction=4, dilation = 1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size[0] * kernel_size[1], 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.sigmoid(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size[0], self.kernel_size[1])
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups, dilation=self.dilation)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x

class Attention(nn.Module):
    def __init__(self,in_planes,K,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.net=nn.Conv2d(in_planes,K,kernel_size=1,bias=False)
        self.sigmoid=nn.Sigmoid()

        if(init_weight):
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return self.sigmoid(att)

class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, K=K, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if (bias):
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if (self.init_weight):
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if (self.bias is not None):
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

class DynamicDWConv_my(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.sigmoid(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x



class concat64_2(nn.Module):
    def __init__(self, in_channel):
        super(concat64_2, self).__init__()
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=(3, 3), stride=1, padding=1)
            # , nn.PReLU()
            # , nn.BatchNorm2d(64)
        )




        self.add_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3), stride=1, padding=1)
            # , nn.PReLU()
            # , nn.BatchNorm2d(64)
        )



        self.concat_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=(3, 3), stride=1, padding=1)
            # , nn.PReLU()
            # , nn.BatchNorm2d(64)
        )

    def forward(self, x, y):
        out1 = torch.cat([x, y], dim=1)
        out1 = self.concat_conv(out1)
        out2 = x + y
        out2 = self.add_conv(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.concat_conv2(out)
        return out



# class concat16_5(nn.Module):
#     def __init__(self):
#             super(concat16_5, self).__init__()
#             self.concat_conv = nn.Sequential(
#                 nn.Conv2d(in_channels=16 * 5, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#                 , nn.PReLU()
#                 # , nn.BatchNorm2d(1)
#             )
#
#             self.mul_conv = nn.Sequential(
#                 nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#                 , nn.PReLU(),
#                 # , nn.BatchNorm2d(1)
#                 # nn.Sigmoid()
#             )
#
#             self.add_conv = nn.Sequential(
#                 nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#                 , nn.PReLU()
#                 # , nn.BatchNorm2d(1)
#             )
#
#             self.concat_conv2 = nn.Sequential(
#                 nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
#                 , nn.PReLU()
#                 # , nn.BatchNorm2d(1)
#             )
#
#
#
#     def forward(self, x1, x2, x3, x4, x5):
#         out1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
#         out1 = self.concat_conv(out1)
#         out2 = x1 + x2 + x3 + x4+ x5
#         out2 = self.add_conv(out2)
#         out3 = x1 * x2 * x3 * x4 * x5
#         out3 = self.mul_conv(out3)
#         out = torch.cat([out1,out2,out3],dim = 1)
#         out = self.concat_conv2(out)
#         return out

class concat16_5(nn.Module):
    def __init__(self):
            super(concat16_5, self).__init__()
            self.concat_conv = nn.Sequential(
                nn.Conv2d(in_channels=16 * 5, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
                , nn.PReLU()
                # , nn.BatchNorm2d(1)
            )

            self.mul_conv = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
                , nn.PReLU(),
                # , nn.BatchNorm2d(1)
                # nn.Sigmoid()
            )

            self.add_conv = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
                , nn.PReLU()
                # , nn.BatchNorm2d(1)
            )

            self.concat_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
                , nn.PReLU()
                # , nn.BatchNorm2d(1)
            )



    def forward(self, x1, x2, x3, x4, x5):
        out1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out1 = self.concat_conv(out1)
        # out2 = x1 + x2 + x3 + x4+ x5
        # out2 = self.add_conv(out2)
        # out3 = x1 * x2 * x3 * x4 * x5
        # out3 = self.mul_conv(out3)
        # out = torch.cat([out1,out2,out3],dim = 1)
        out = self.concat_conv2(out1)
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)  # depthwise conv
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=(9,3), padding=(4,1), groups=dim)  # depthwise conv
        # self.dwconv1 = DynamicDWConv(dim, kernel_size=(11,3), stride=1, padding=(5,1), groups=dim)
        # self.dwconv2 = DynamicDWConv(dim, kernel_size=(3,11), stride=1, padding=(1,5), groups=dim)
        # self.dwconv = CondConv(in_planes=dim, out_planes=dim, kernel_size=7, stride=1, padding=3, grounps= dim,bias=True)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=(3,9), padding=(1,4), groups=dim)  # depthwise conv
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=(1, 1))  # depthwise conv
        # self.conv3 = DynamicDWConv(dim, kernel_size=(1,1), stride=1, padding=(0,0)) # depthwise conv
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=(1, 1),groups=dim)
        # self.conv4 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1,1))
        # self.dwconv1_1 = nn.Conv2d(dim, dim, kernel_size=(11, 3), padding=(5, 1), groups=dim)  # depthwise conv
        # self.dwconv1 = DynamicDWConv(dim, kernel_size=(11,3), stride=1, padding=(5,1), groups=dim)
        # self.dwconv2 = DynamicDWConv(dim, kernel_size=(3,11), stride=1, padding=(1,5), groups=dim)
        # self.dwconv = CondConv(in_planes=dim, out_planes=dim, kernel_size=7, stride=1, padding=3, grounps= dim,bias=True)
        # self.dwconv2_1 = nn.Conv2d(dim, dim, kernel_size=(3, 11), padding=(1, 5), groups=dim)  # depthwise conv
        # self.conv3_1 = nn.Conv2d(dim, dim, kernel_size=(1, 1))  # depthwise conv

        # self.dwconv2 = DynamicDWConv(dim, kernel_size=9, stride=1, padding=4, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None


        # self.split_list = (self.sc, self.sc, self.sc, dim - self.sc * 3 )
    def forward(self, x):
        input = x
        # x = self.dwconv1(x)
        # x = self.dwconv2(x)

        x = self.dwconv1(x) + self.dwconv2(x) + self.conv3(x)
        # x = self.dwconv1(x)  + self.conv4(x)
        x= self.conv4(x)
        x = self.dwconv3(x)
        # x = s# elf.dwconv1_1(x)
        # + self.dwconv2_1(x) + self.conv3_1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class Blockd(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)  # depthwise conv
        self.dwconv1 = DynamicDWConv(dim, kernel_size=(9,9), stride=1, padding=(4,4), groups=dim)
        # self.dwconv = CondConv(in_planes=dim, out_planes=dim, kernel_size=7, stride=1, padding=3, grounps= dim,bias=True)
        # self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)  # depthwise conv
        self.dwconv2 = DynamicDWConv(dim, kernel_size=(9,9), stride=1, padding=(4,4), groups=dim)
        # self.conv3 = nn.Conv2d(dim, dim, kernel_size=(1, 1), groups=dim)  # depthwise conv
        # self.conv4 = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        # self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)  # depthwise conv



        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None


    def forward(self, x):
        input = x
        x = self.dwconv1(x)

        x = self.dwconv2(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

# without bn version


class convnextSingle(nn.Module):
    def __init__(self, base, path=None, visualize = False):
        super(convnextSingle, self).__init__()
        if base not in ["vgg16", "resnet50", "PVT_v2","convnext"]:
            raise NotImplementedError("The argument 'base' in backbone is False")
        if base == "convnext":
            self.backbone1,self.backbone_dims = convnext_base(pretrained=True,in_22k = True)
            # self.backbone2, self.backbone_dims = convnext_base(pretrained=True, in_22k=False)

            filters = [64] + self.backbone_dims
        if base == "resnet50":
            self.backbone1 = Backbone_ResNet50_in3(pretrained=True)
            self.backbone2 = Backbone_ResNet50_in3(pretrained=True)
            self.backbone_dims = [64, 256, 512, 1024, 2048]
            filters =self.backbone_dims

        if base == "vgg16":
            self.backbone1 = Backbone_vgg16_in3(pretrained=True)
            self.backbone2 = Backbone_vgg16_in3(pretrained=True)
            self.backbone_dims = [64, 256, 512, 512, 512]
            filters =self.backbone_dims

        self.visualize = visualize
        self.downsample_layer = nn.Sequential(
            LayerNorm(3, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(3, 64, kernel_size=(2,2), stride=2),
            Block(64),
            Block(64),
            # Block(64),
            # Block(64),
        )

        self.CatChannels = filters[0]
        self.CatBlocks = 1
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        # self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        # self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        # self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h1_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd4_block = nn.Sequential(
            # LayerNorm(filters[0], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(8, 8, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(8, 8, ceil_mode=True),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # h2->160*160, hd4->40*40, Pooling 4 times
        # self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        # self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        # self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h2_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd4_block = nn.Sequential(
            # LayerNorm(filters[1], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(4, 4, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(4, 4, ceil_mode=True),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )


        # h3->80*80, hd4->40*40, Pooling 2 times
        # self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        # self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h3_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h3_PT_hd4_block = nn.Sequential(
            # LayerNorm(filters[2], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(2, 2, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # h4->40*40, hd4->40*40, Concatenation
        # self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        # self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)
        self.h4_Cat_hd4_block = nn.Sequential(
            # nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )


        # hd5->20*20, hd4->40*40, Upsample 2 times
        # self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        # self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        # self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)
        self.hd5_UT_hd4_block = nn.Sequential(
            # LayerNorm(filters[4], eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14


            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels * 5, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)
        # self.fusion4_block = nn.Sequential(
        #     Block(self.UpChannels),
        # )

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        # self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        # self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        # self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h1_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd3_block = nn.Sequential(
            # LayerNorm(filters[0], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(4, 4, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(4, 4, ceil_mode=True),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # h2->160*160, hd3->80*80, Pooling 2 times
        # self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        # self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h2_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd3_block = nn.Sequential(
            # LayerNorm(filters[1], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(2, 2, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # h3->80*80, hd3->80*80, Concatenation
        # self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        # self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)
        self.h3_Cat_hd3_block = nn.Sequential(
            # nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # hd4->40*40, hd4->80*80, Upsample 2 times
        # self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        # self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd3_block = nn.Sequential(
            # LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14


            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # hd5->20*20, hd4->80*80, Upsample 4 times
        # self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        # self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        # self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)
        self.hd5_UT_hd3_block = nn.Sequential(
            # LayerNorm(filters[4], eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=4, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 14*14


            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels * 5, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)
        # self.fusion3_block = nn.Sequential(
        #     Block(self.UpChannels),
        # )

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        # self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        # self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h1_PT_hd2_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd2_block = nn.Sequential(
            # LayerNorm(filters[0], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(2, 2, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.MaxPool2d(2, 2, ceil_mode=True),


            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # h2->160*160, hd2->160*160, Concatenation
        # self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        # self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)
        self.h2_Cat_hd2_block = nn.Sequential(
            # LayerNorm(filters[1], eps=1e-6, data_format="channels_first"),
            # nn.MaxPool2d(2, 2, ceil_mode=True),
            # nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # hd3->80*80, hd2->160*160, Upsample 2 times
        # self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        # self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd2_block = nn.Sequential(
            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),
            Block(self.CatChannels),
            # Block(self.CatChannels),
            Block(self.CatChannels),
        )

        # hd4->40*40, hd2->160*160, Upsample 4 times
        # self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        # self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd2_block = nn.Sequential(
            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),
            Block(self.CatChannels),
            # Block(self.CatChannels),
            Block(self.CatChannels),
        )

        # hd5->20*20, hd2->160*160, Upsample 8 times
        # self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        # self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        # self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd5_UT_hd2_block = nn.Sequential(
            # LayerNorm(filters[4], eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=8, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),

            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=8, mode='bilinear'),  # 14*14

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels * 5, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)
        # self.fusion2_block = nn.Sequential(
        #     Block(self.UpChannels),
        # )
        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        # self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        # self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        # self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)
        self.h1_Cat_hd1_block = nn.Sequential(
            # nn.Conv2d(in_channels=filters[0], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),

            Block(self.CatChannels),
            # Block(self.CatChannels),
            Block(self.CatChannels),
        )

        # hd2->160*160, hd1->320*320, Upsample 2 times
        # self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        # self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd2_UT_hd1_block = nn.Sequential(
            # LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),

            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14

            Block(self.CatChannels),
            # Block(self.CatChannels),
            Block(self.CatChannels),
        )

        # hd3->80*80, hd1->320*320, Upsample 4 times
        # self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        # self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd1_block = nn.Sequential(
            # LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=4, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),
            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 14*14

            Block(self.CatChannels),
            # Block(self.CatChannels),
            Block(self.CatChannels),
        )

        # hd4->40*40, hd1->320*320, Upsample 8 times
        # self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        # self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        # self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd1_block = nn.Sequential(
            # LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=8, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.UpChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),

            LayerNorm(self.UpChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=8, mode='bilinear'),  # 14*14


            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # hd5->20*20, hd1->320*320, Upsample 16 times
        # self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        # self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        # self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        # self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd5_UT_hd1_block = nn.Sequential(
            # LayerNorm(filters[4], eps=1e-6, data_format="channels_first"),
            # nn.Upsample(scale_factor=16, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1,
            #           padding=0),

            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=16, mode='bilinear'),  # 14*14

            Block(self.CatChannels),
            Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels * 5, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)
        # self.fusion1_block = nn.Sequential(
        #     Block(self.UpChannels),
        # )

        # -------------Bilinear Upsampling--------------


        # self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        # self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')




        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        # self.outconv5 = nn.Conv2d(filters[4], 1, 3, padding=1)

        self.outconv5 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)

        self.outconv0 = nn.Sequential(
            LayerNorm(self.CatChannels, eps=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14*14
            # nn.Conv2d(in_channels=self.CatChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            # nn.Conv2d(in_channels=self.CatChannels, out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Block(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
            nn.Conv2d(in_channels=self.CatChannels, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
        )

        # self.cat4 = concat64_5(self.CatChannels)
        # self.cat3 = concat64_5(self.CatChannels)
        # self.cat2 = concat64_5(self.CatChannels)
        # self.cat1 = concat64_5(self.CatChannels)

        self.get_feature2 = nn.Sequential(
            nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature3 = nn.Sequential(
            nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature4 = nn.Sequential(
            nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature5 = nn.Sequential(
            nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        self.get_feature2_d = nn.Sequential(
            nn.Conv2d(in_channels=filters[1], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature3_d = nn.Sequential(
            nn.Conv2d(in_channels=filters[2], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature4_d = nn.Sequential(
            nn.Conv2d(in_channels=filters[3], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )
        self.get_feature5_d = nn.Sequential(
            nn.Conv2d(in_channels=filters[4], out_channels=self.CatChannels, kernel_size=(1, 1), stride=1, padding=0),
            Blockd(self.CatChannels),
            # Block(self.CatChannels),
            # Block(self.CatChannels),
        )

        # self.fuse2 = concat64_2(64)
        # self.fuse3 = concat64_2(64)
        # self.fuse4 = concat64_2(64)
        # self.fuse5 = concat64_2(64)

        self.create_weight1 = nn.Sequential(
            nn.Conv2d(filters[0], 64, 1, bias=False),
            Block(64),
            # Block(64),
            # Block(64),
            # nn.BatchNorm2d(64),
            #
            # nn.Sigmoid(),
            nn.Conv2d(64, 16, 1, bias=False),
        )

        self.create_weight2 = nn.Sequential(
            nn.Conv2d(filters[1], 64, 1, bias=False),
            Block(64),
            # Block(64),
            # Block(64),
            # nn.BatchNorm2d(64),
            #
            # nn.Sigmoid(),
            nn.Conv2d(64, 16, 1, bias=False),
        )

        self.create_weight3 = nn.Sequential(
            nn.Conv2d(filters[2], 64, 1, bias=False),
            Block(64),
            # Block(64),
            # Block(64),
            # nn.BatchNorm2d(64),
            #
            # nn.Sigmoid(),
            nn.Conv2d(64, 16, 1, bias=False),
        )

        self.create_weight4 = nn.Sequential(
            nn.Conv2d(filters[3], 64, 1, bias=False),
            Block(64),
            # Block(64),
            # Block(64),
            # nn.BatchNorm2d(64),
            #
            # nn.Sigmoid(),
            nn.Conv2d(64, 16, 1, bias=False),
        )

        self.create_weight5 = nn.Sequential(
            nn.Conv2d(filters[4], 64, 1, bias=False),
            Block(64),
            # Block(64),
            # Block(64),
            # nn.BatchNorm2d(64),
            #

            # nn.Sigmoid(),
            nn.Conv2d(64, 16, 1, bias=False),
        )

        self.upattn6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upattn5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upattn4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upattn3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upattn2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.fuse_attn = concat16_5()

        self.create_attn = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                      padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                      padding=1),
            nn.Conv2d(64, 3, 1, bias=False),
        )
        self.fuse1 = concat64_2(64)
        self.fuse2 = concat64_2(64)
        self.fuse3 = concat64_2(64)
        self.fuse4 = concat64_2(64)
        self.fuse5 = concat64_2(64)

    def encoder_help(self, encoder, x):  ## 注意foward()
        B = x.size()[0]
        e, h, w = encoder[0](x)
        for enco in encoder[1]:
            e = enco(e, h, w)
        e = encoder[2](e)
        e = e.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return e

    def forward(self, x, depth):

        # print(x.size())  #[B,3,224,224]
        #将输入传入Encoder的第一个stage
        # e1 = self.encoder_help(self.encoder1, x)   # 4x [B,64,56,56]
        # # #将第一个stage的Tensor转换回特征图
        # # B, N, C = f1.shape
        # # H = N ** 0.5
        # # W = N ** 0.5
        # # e1 = x.transpose(1, 2).view(B, C, H, W)
        # e2 = self.encoder_help(self.encoder2, e1)  # 2x [B,128,28,28]
        # e3 = self.encoder_help(self.encoder3, e2)  # 2x [B,320,14,14]
        # e4 = self.encoder_help(self.encoder4, e3)  # 2x [B,512,7,7]
        e2,e3,e4,e5 = self.backbone1(x)
        e1 = self.downsample_layer(x)

        weight_depth1 = self.create_weight1(e1)
        weight_depth2 = self.create_weight2(e2)
        weight_depth3 = self.create_weight3(e3)
        weight_depth4 = self.create_weight4(e4)
        weight_depth5 = self.create_weight5(e5)

        weight_depth2 = self.upattn2(weight_depth2)
        weight_depth3 = self.upattn3(weight_depth3)
        weight_depth4 = self.upattn4(weight_depth4)
        weight_depth5 = self.upattn5(weight_depth5)

        edge1 = self.up2(weight_depth1[:, 0:1, :, :])
        edge2 = self.up2(weight_depth2[:, 0:1, :, :])
        edge3 = self.up2(weight_depth3[:, 0:1, :, :])

        weight_depth = self.create_attn(
            self.fuse_attn(weight_depth1, weight_depth2, weight_depth3, weight_depth4, weight_depth5))

        caliDepth = depth * weight_depth


        ed2, ed3, ed4, ed5 = self.backbone1(caliDepth)
        ed1 = self.downsample_layer(caliDepth)
        # weight_depth1 = self.create_weight1(e1)
        # weight_depth2 = self.create_weight2(e2)
        # weight_depth3 = self.create_weight3(e3)
        # weight_depth4 = self.create_weight4(e4)
        # weight_depth5 = self.create_weight5(e5)
        #
        # weight_depth2 = self.upattn2(weight_depth2)
        # weight_depth3 = self.upattn3(weight_depth3)
        # weight_depth4 = self.upattn4(weight_depth4)
        # weight_depth5 = self.upattn5(weight_depth5)
        #
        # weight_depth = self.create_attn(self.fuse_attn(weight_depth1, weight_depth2, weight_depth3, weight_depth4, weight_depth5))
        #
        # caliDepth = depth * weight_depth
        # caliDepth = depth

        # ed2, ed3, ed4, ed5 = self.backbone1(caliDepth)


            # weight_depth = self.create_weight(dhd5)






        # h2 = h2 + dh2
        # h3 = h3 + dh3
        # h4 = h4 + dh4
        # hd5 = hd5 + dhd5
        # h1 = e1 + d1
        # h2 = self.get_feature2(e2) + self.get_feature2_d(d2)
        # h3 = self.get_feature3(e3) + self.get_feature3_d(d3)
        # h4 = self.get_feature4(e4) + self.get_feature4_d(d4)
        # hd5 = self.get_feature5(e5) + self.get_feature5_d(d5)
        h1 = self.fuse1(e1, ed1)
        h2 = self.fuse2(self.get_feature2(e2), self.get_feature2_d(ed2))
        h3 = self.fuse3(self.get_feature3(e3), self.get_feature3_d(ed3))
        h4 = self.fuse4(self.get_feature4(e4), self.get_feature4_d(ed4))
        hd5 = self.fuse5(self.get_feature5(e5), self.get_feature5_d(ed5))

        # h2 = self.fuse2(self.get_feature2(e2),self.get_feature2_d(d2))
        # h3 = self.fuse3(self.get_feature3(e3),self.get_feature3_d(d3))
        # h4 = self.fuse4(self.get_feature4(e4),self.get_feature4_d(d4))
        # hd5 = self.fuse5(self.get_feature5(e5),self.get_feature5_d(d5))


        ## -------------Decoder-------------
        # h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h1_PT_hd4 = self.h1_PT_hd4_block(h1)

        # h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h2_PT_hd4 = self.h2_PT_hd4_block(h2)

        # h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h3_PT_hd4 = self.h3_PT_hd4_block(h3)


        # h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        h4_Cat_hd4 = self.h4_Cat_hd4_block(h4)

        # hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd5_UT_hd4 = self.hd5_UT_hd4_block(hd5)

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        # hd4 = self.fusion4_block(
        #     torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)) # hd4->40*40*UpChannels

        # hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
        #     self.cat4(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4))))  # hd4->40*40*UpChannels


        # h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        # h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        # h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        # hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        # hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))

        h1_PT_hd3 = self.h1_PT_hd3_block(h1)
        h2_PT_hd3 = self.h2_PT_hd3_block(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3_block(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3_block(hd4)
        hd5_UT_hd3 = self.hd5_UT_hd3_block(hd5)

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        # hd3 = self.fusion3_block(
        #     torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))  # hd3->80*80*UpChannels

        # hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
        # self.cat3(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3))))  # hd3->80*80*UpChannels

        # h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        # h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        # hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        # hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        # hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))

        h1_PT_hd2 = self.h1_PT_hd2_block(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2_block(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2_block(hd3)
        hd4_UT_hd2 = self.hd4_UT_hd2_block(hd4)
        hd5_UT_hd2 = self.hd5_UT_hd2_block(hd5)

        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        # hd2 = self.fusion2_block(
        #     torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))  # hd2->160*160*UpChannels

        # hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
        #     self.cat2(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2))))  # hd2->160*160*UpChannels

        # h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        # hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        # hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        # hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        # hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))

        h1_Cat_hd1 = self.h1_Cat_hd1_block(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1_block(hd2)
        hd3_UT_hd1 = self.hd3_UT_hd1_block(hd3)
        hd4_UT_hd1 = self.hd4_UT_hd1_block(hd4)
        hd5_UT_hd1 = self.hd5_UT_hd1_block(hd5)

        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        # hd1 = self.fusion1_block(
        #     torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))  # hd1->320*320*UpChannels

        # hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
        #     self.cat1(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore6(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore5(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore4(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore3(d2) # 128->256

        d1 = self.outconv1(hd1) # 256
        d1 = self.upscore2(d1)
        d0 = self.outconv0(hd1)

        print_dict = {}
        if self.visualize == True:
            # print_dict = dict(h1=h1.sigmoid(),
            #                   h2=h2.sigmoid(),
            #                   h3=h3.sigmoid(),
            #                   h4=h4.sigmoid(),
            #                   hd5=hd5.sigmoid()
            #                   )

            print_dict = dict(
                              depth = depth.sigmoid(),
                              caliDepth = caliDepth.sigmoid(),
                              e1=e1.sigmoid(),
                              e2=e2.sigmoid(),
                              e3=e3.sigmoid(),
                              e4=e4.sigmoid(),
                              e5=e5.sigmoid(),
                              edge1=edge1.sigmoid(),
                              edge2=edge2.sigmoid(),
                              edge3=edge3.sigmoid(),

                              h2=h2.sigmoid(),
                              h3=h3.sigmoid(),
                              h4=h4.sigmoid(),
                              hd5=hd5.sigmoid(),
                              hd1=h1.sigmoid(),
                              hd2=hd2.sigmoid(),
                              hd3=hd3.sigmoid(),
                              hd4=hd4.sigmoid(),
                              d1=d1.sigmoid(),
                              d2=d2.sigmoid(),
                              d3=d3.sigmoid(),
                              d4=d4.sigmoid(),
                              d5=d5.sigmoid(),
                              )

        heatmap_dict = dict(
            weight_depth = weight_depth.sigmoid(),
        )
        # d0 = (d1 + d2 + d3 + d4 + d5) / 5

        return d0, d1, d2, d3, d4, d5,edge1,edge2,edge3, print_dict, heatmap_dict


class convnext(nn.Module):
    def __init__(self, base, path=None, visualize = False):
        super(convnext, self).__init__()
        if base not in ["vgg16", "resnet50", "convnext"]:
            raise NotImplementedError("The argument 'base' in backbone is False")
        self.visualize = visualize

        self.single = convnextSingle(base = base, visualize = self.visualize)
        self.outConv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=1,padding=1),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
        )

        self.final_fuse = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        filters = [64, 128, 256, 512, 1024]


        # self.my_cat = concat3_2()

        # self.final_fuse = concat1_5()
    def encoder_help(self, encoder, x):  ## 注意foward()
        B = x.size()[0]
        e, h, w = encoder[0](x)
        for enco in encoder[1]:
            e = enco(e, h, w)
        e = encoder[2](e)
        e = e.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return e
    def forward(self, x,depth):
        depth = torch.cat([depth, depth, depth], dim = 1)
        d0, d1, d2, d3, d4, d5,edge1,edge2,edge3, print_dict, heatmap_dict = self.single(x, depth)










        pre = [d0.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(),edge1.sigmoid(),edge2.sigmoid(),edge3.sigmoid()]

        # pre = self.final_fuse(d1, d2, d3, d4, d5)

        # if arg["mid_output"]==True:
        name_dict=[]
        value_dict=[]
        heatmap_name_dict = []
        heatmap_value_dict = []

            # print_dict=dict(
            #     e0=e0.sigmoid(),d0=d0.sigmoid(),e1=e1.sigmoid(),ed1=ed1.sigmoid(),efuse1=efuse1.sigmoid(),e2=e2.sigmoid(),ed2=ed2.sigmoid(),efuse2=efuse2.sigmoid(),
            #     e3=e3.sigmoid(),ed3=ed3.sigmoid(),efuse3=efuse3.sigmoid(),pre=pre.sigmoid())

            # print_dict = dict(
            #     e1=e1.sigmoid(),e2=e2.sigmoid(),e3=e3.sigmoid(),e4=e4.sigmoid(),
            #     ed1=ed1.sigmoid(), ed2=ed2.sigmoid(), ed3=ed3.sigmoid(), ed4=ed4.sigmoid(),
            #     efuse1=efuse1.sigmoid(),efuse2=efuse2.sigmoid(),efuse3=efuse3.sigmoid(),efuse4=efuse4.sigmoid(),
            #     d0=d0.sigmoid(), d1=d1.sigmoid(),d2=d2.sigmoid(), d3=d3.sigmoid(),d4=d4.sigmoid(),
            #     pre=pre.sigmoid())
            # print_dict = dict(
            #     # e32_64=e32_64.sigmoid(),
            #     d1=d1.sigmoid(),
            #     pre=pre.sigmoid())

        for key,value in heatmap_dict.items():
            heatmap_name_dict.append(key)
            heatmap_value_dict.append(value)



        for key,value in print_dict.items():
            name_dict.append(key)
            value_dict.append(value)
        return value_dict,name_dict,heatmap_value_dict,heatmap_name_dict,pre



if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 1, 1)

    # net = PVTUNet(base=arg["backbone"],path=arg["pretrained"])
    #
    # out = net(rgb)
    print(rgb)
    print(rgb[:, 0:1,:, :])
    print(rgb[:,1:2,:,:])
    print(rgb[:,2:3,:,:])