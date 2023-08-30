import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from models.generator.baseModule import RCCAModule, GCFF


#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'elu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'bilinear')
        x = self.conv2d(x)
        return x

#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'elu', norm = 'none', sn = False, return_mask=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
        self.return_mask = return_mask

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        if self.activation:
            conv = self.activation(conv)
        x = conv * gated_mask
        if self.return_mask:
            return x, gated_mask
        else:
            return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'bilinear')
        x = self.gated_conv2d(x)
        return x

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
"""
class GatedRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedRes, self).__init__()
        self.conv = nn.Sequential(
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn'),
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        )
    def forward(self, x):
        y = self.conv(x)
        return x + y
"""

class GatedRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedRes, self).__init__()
        self.conv_global = nn.Sequential(
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', dilation=1, activation='lrelu', norm='none'),
            GatedConv2d(in_channels, out_channels, 3, 1, 2, pad_type='zero', dilation=2, activation='lrelu', norm='none'),
            GatedConv2d(in_channels, out_channels, 3, 1, 4, pad_type='zero', dilation=4, activation='lrelu', norm='none'),
            GatedConv2d(in_channels, out_channels, 3, 1, 8, pad_type='zero', dilation=8, activation='lrelu', norm='in'),
        )
        self.conv_local = nn.Sequential(
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='none'),
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in')
        )
    def forward(self, x):
        y = self.conv_local(x) + x
        z = self.conv_global(y) + y
        return z + x


class gatedRes(nn.Module):
    def __init__(self, ch):
        super(gatedRes, self).__init__()
        self.conv_g0 = GatedConv2d(ch, ch, 3, 1, 1, pad_type='zero', dilation=1, activation='lrelu', norm='in')
        self.w_g0 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv_g1 = GatedConv2d(ch, ch, 3, 1, 2, pad_type='zero', dilation=2, activation='lrelu', norm='in')
        self.w_g1 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv_g2 = GatedConv2d(ch, ch, 3, 1, 4, pad_type='zero', dilation=4, activation='lrelu', norm='in')
        self.w_g2 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv_g3 = GatedConv2d(ch, ch, 3, 1, 8, pad_type='zero', dilation=8, activation='lrelu', norm='in')
        self.w_g3 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv_local = nn.Sequential(
            GatedConv2d(ch, ch, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in'),
            GatedConv2d(ch, ch, 3, 1, 1, pad_type='zero', activation='lrelu', norm='in')
        )
    def forward(self, x):
        y = self.conv_local(x) + x
        z0 = self.conv_g0(y) * self.w_g0(y) + y
        z1 = self.conv_g1(z0) * self.w_g1(y) + z0
        z2 = self.conv_g2(z1) * self.w_g2(y) + z1
        z = self.conv_g3(z2) * self.w_g3(y) + z2
        return z + y

class Gated_Atten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gated_Atten, self).__init__()
        self.conv_global = RCCAModule(in_channels, in_channels)
        self.conv_local = nn.Sequential(
            GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn'),
            GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        )
        self.gcff = GCFF(in_channels, in_channels)
        self.conv_out = GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        l = self.conv_local(x)
        g = self.conv_global(x)
        out = self.gcff(l, g)
        out = self.conv_out(out)
        return out
"""
class skipDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode='none', out=False):
        super(skipDecoder, self).__init__()
        self.mode = mode
        self.out = out
        self.conv1 = GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        self.conv2 = GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        self.res = GatedRes(in_channels, in_channels)
        self.conv_atten = Gated_Atten(in_channels, out_channels)
        if self.mode == 'upsample':
            self.transconv = TransposeGatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')

        if self.out is True:
            self.activation = nn.Tanh()

    def forward(self, x, x_skip):
        y = self.conv1(x)
        z = self.res(x_skip+y)
        out = self.conv2(z+y)
        if self.mode == "upsample":
            out = self.transconv(out)
        out = self.conv_atten(out)
        if self.out is True:
            out = self.activation(out)
        return out
"""

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

class skipDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode='none', out=False, ifnoise=False):
        super(skipDecoder, self).__init__()
        self.mode = mode
        self.out = out
        self.conv1 = GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        self.conv2 = GatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        # self.res = GatedRes(in_channels, in_channels)
        self.res = gatedRes(in_channels)
        if self.mode == 'upsample':
            self.transconv = TransposeGatedConv2d(in_channels, in_channels, 3, 1, 1, pad_type='zero',
                                                  activation='lrelu', norm='bn')
            if self.out is True:
                self.conv3 = nn.Sequential(
                    GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='none', norm='bn'),
                    nn.Tanh()
                )
            else:
                self.conv3 = GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        self.ifnoise = ifnoise
        if ifnoise:
            self.noise = nn.Sequential(
                NoiseInjection(),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x, x_skip):
        y = self.conv1(x)
        z = self.res(x_skip+y)
        out = self.conv2(z+y)
        if self.mode == "upsample":
            out = self.transconv(out)
            out = self.conv3(out)
        if self.ifnoise:
            out = self.noise(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            GatedConv2d(in_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn'),
            TransposeGatedConv2d(out_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn'),
            GatedConv2d(out_channels, out_channels, 3, 1, 1, pad_type='zero', activation='lrelu', norm='bn')
        )
    def forward(self, x):
        x = self.conv(x)
        return x
