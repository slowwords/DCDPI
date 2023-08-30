import torch
import torch.nn as nn
import torch.nn.functional as F
from models.generator.pconv import PConvBNActiv
from models.generator.baseModule import SENet, MCFF, RCCAModule, MultiHeadAttention
from models.generator.gconv import GatedConv2d, Decoder
from models.generator.gconv import skipDecoder
from options.base_options import BaseOptions

opt = BaseOptions().parse()

class FeatureExtract(nn.Module):

    def __init__(self, opt):
        super(FeatureExtract, self).__init__()

        self.ec_1 = nn.Sequential(
            GatedConv2d(opt.in_channels + 1, opt.latent_channels, 5, 1, 2, pad_type='zero',
                        activation='lrelu', norm='none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels, 5, 2, 2, pad_type='zero',  # 128
                        activation='lrelu', norm='none')
        )
        self.ec_2 = nn.Sequential(
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 1, 1, pad_type='zero',
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 2, 1, pad_type='zero',  # 64
                        activation='lrelu', norm='bn')
        )
        self.ec_3 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type='zero',
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 2, 1, pad_type='zero',  # 32
                        activation='lrelu', norm='bn')
        )
        self.ec_4 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 6, 3, 1, 1, pad_type='zero',
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 6, opt.latent_channels * 6, 3, 2, 1, pad_type='zero',  # 16
                        activation='lrelu', norm='bn')
        )
        self.ec_5 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 6, opt.latent_channels * 8, 3, 1, 1, pad_type='zero',
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 2, 1, pad_type='zero',  # 8
                        activation='lrelu', norm='bn')
        )
        self.ec_6 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 1, pad_type='zero',
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 2, 1, pad_type='zero',  # 4
                        activation='lrelu', norm='bn')
        )

    def forward_once(self, image):
        ec = {}
        ec['ec_1'] = self.ec_1(image)
        ec['ec_2'] = self.ec_2(ec['ec_1'])
        ec['ec_3'] = self.ec_3(ec['ec_2'])
        ec['ec_4'] = self.ec_4(ec['ec_3'])
        ec['ec_5'] = self.ec_5(ec['ec_4'])
        ec['ec_6'] = self.ec_6(ec['ec_5'])

        return ec

    def forward(self, input_image, input_edge, mask):
        image_in = torch.cat((input_image, mask), dim=1)
        edge_in = torch.cat((input_edge, mask, mask), dim=1)

        ec_image = self.forward_once(image_in)
        ec_edge = self.forward_once(edge_in)

        return ec_image, ec_edge

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.encoder = FeatureExtract(opt)

        self.mcff = MCFF(opt.latent_channels * 8, opt.latent_channels * 8)

        # res block
        self.ec_out = nn.Sequential(
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 1, pad_type='zero',  # 4
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 2, dilation=2, pad_type='zero',  # 4
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 4, dilation=4, pad_type='zero',  # 4
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 8, dilation=8, pad_type='zero',  # 4
                        activation='lrelu', norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 16, dilation=16, pad_type='zero',  # 4
                        activation='lrelu', norm='bn'),
            SENet(opt.latent_channels * 8),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 1, pad_type='zero', activation='lrelu',
                        norm='bn'),
            GatedConv2d(opt.latent_channels * 8, opt.latent_channels * 8, 3, 1, 1, pad_type='zero', activation='lrelu',
                        norm='bn')
        )

        self.dc_6 = skipDecoder(opt.latent_channels * 8, opt.latent_channels * 8, mode='upsample', ifnoise=True)
        self.dc_5 = skipDecoder(opt.latent_channels * 8, opt.latent_channels * 6, mode='upsample', ifnoise=True)
        self.dc_4 = skipDecoder(opt.latent_channels * 6, opt.latent_channels * 4, mode='upsample', ifnoise=True)
        self.dc_3 = skipDecoder(opt.latent_channels * 4, opt.latent_channels * 2, mode='upsample', ifnoise=True)
        self.dc_2 = skipDecoder(opt.latent_channels * 2, opt.latent_channels, mode='upsample', ifnoise=True)
        self.dc_1 = skipDecoder(opt.latent_channels, opt.out_channels, mode='upsample', out=True)


        self.skip_6 = RCCAModule(opt.latent_channels * 8, opt.latent_channels * 8, multi_heads=0)
        self.skip_5 = RCCAModule(opt.latent_channels * 8, opt.latent_channels * 8, multi_heads=0)
        self.skip_4 = RCCAModule(opt.latent_channels * 6, opt.latent_channels * 6, multi_heads=0)
        """
        self.skip_6 = nn.Sequential(
            MultiHeadAttention(opt.latent_channels * 8, num_heads=1),
            MultiHeadAttention(opt.latent_channels * 8, num_heads=1)
        )
        self.skip_5 = nn.Sequential(
            MultiHeadAttention(opt.latent_channels * 8, num_heads=1),
            MultiHeadAttention(opt.latent_channels * 8, num_heads=1)
        )
        self.skip_4 = nn.Sequential(
            MultiHeadAttention(opt.latent_channels * 6, num_heads=1),
            MultiHeadAttention(opt.latent_channels * 6, num_heads=1)
        )"""
        self.show_ec1 = nn.Conv2d(opt.latent_channels, 3, 1, 1, 0)
        self.show_dc3 = nn.Conv2d(opt.latent_channels * 2, 3, 1, 1, 0)
        self.show_dc2 = nn.Conv2d(opt.latent_channels, 3, 1, 1, 0)


    def forward(self, input_image, input_edge, mask):

        ec_image, ec_edge = self.encoder(input_image, input_edge, mask)
        fusion = self.mcff(ec_image['ec_6'], ec_edge['ec_6'])
        ec_image['out'] = self.ec_out(fusion)

        dc = {}
        from IPython import embed
        # embed()
        dc['dc_6'] = self.dc_6(ec_image['out'], self.skip_6(ec_image['ec_6']))  # torch.Size([1, 512, 8, 8])

        dc['dc_5'] = self.dc_5(dc['dc_6'], self.skip_5(ec_image['ec_5']))  # torch.Size([1, 512, 16, 16])
        dc['dc_4'] = self.dc_4(dc['dc_5'], self.skip_4(ec_image['ec_4']))
        dc['dc_3'] = self.dc_3(dc['dc_4'], ec_image['ec_3'])
        dc['dc_2'] = self.dc_2(dc['dc_3'], ec_image['ec_2'])
        dc['dc_1'] = self.dc_1(dc['dc_2'], ec_image['ec_1'])

        out = dc['dc_1']

        # RGB = {}
        # RGB[f'8x8'] = self.to_rgb6(dc['dc_6'])
        # RGB[f'16x16'] = self.to_rgb5(dc['dc_5'])
        ec1 = self.show_ec1(ec_image['ec_1'])
        edge1 = self.show_ec1(ec_edge['ec_1'])
        dc3 = self.show_dc3(dc['dc_3'])
        dc2 = self.show_dc2(dc['dc_2'])

        return out, ec1, edge1, dc3, dc2


