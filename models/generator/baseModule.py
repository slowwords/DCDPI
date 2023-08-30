import torch
import torch.nn as nn
from options.base_options import BaseOptions

opt = BaseOptions().parse()
device = torch.device(opt.device)

class SENet(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//ratio, False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size, channels, _, _ = images.size()
        avg = self.avg_pool(images).view(batch_size, channels)
        fc = self.fc(avg).view(batch_size, channels, 1, 1)
        return images * fc.expand_as(images)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


def INF(B, H, W):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
    else:
        return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize*width, height, height)
        att_W = concate[:, :, :, height:height+width].contiguous().view(m_batchsize*height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma*(out_H + out_W) + x


class MultiheadCrissCrossAttention(nn.Module):
    """ Multihead Criss-Cross Attention Module"""

    def __init__(self, in_dim, num_heads):
        super(MultiheadCrissCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)

        self.out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        num_heads = self.num_heads
        head_dim = self.head_dim

        proj_query = self.query_conv(x).view(batch_size, channels, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, channels, -1)
        proj_value = self.value_conv(x).view(batch_size, channels, -1)

        proj_query_H = proj_query.view(batch_size, width, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, width, -1)
        proj_query_W = proj_query.view(batch_size, height, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, height, -1)

        proj_key_H = proj_key.view(batch_size, width, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, width, -1)
        proj_key_W = proj_key.view(batch_size, height, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, height, -1)

        proj_value_H = proj_value.view(batch_size, width, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, width, -1)
        proj_value_W = proj_value.view(batch_size, height, -1, channels).permute(0, 3, 1, 2).contiguous().view(
            batch_size * num_heads, channels // num_heads, height, -1)

        energy_H = (torch.bmm(proj_query_H, proj_key_H.permute(0, 1, 3, 2)) + self.INF * torch.eye(width,
                                                                                                   device=x.device)).view(
            batch_size, num_heads, channels // num_heads, width, width).permute(0, 2, 1, 3, 4)
        energy_W = torch.bmm(proj_query_W, proj_key_W.permute(0, 1, 3, 2)).view(batch_size, num_heads,
                                                                                channels // num_heads, height, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 4))
        att_H = concate[:, :, :, :, 0:width].permute(0, 2, 1, 3, 4).contiguous().view(batch_size * num_heads,
                                                                                      channels // num_heads, width,
                                                                                      width)
        att_W = concate[:, :, :, :, width:width + height].contiguous().view(batch_size * num_heads,
                                                                            channels // num_heads, height, width)

        out_H = torch.bmm(proj_value_H.view(batch_size * num_heads, channels // num_heads, width, -1),
                          att_H.permute(0, 2, 1)).view(batch_size, num_heads, channels // num_heads, width,
                                                       height).permute(0, 2, 1, 3, 4)
        out_W = torch.bmm(proj_value_W.view(batch_size * num_heads, channels // num_heads, height, -1),
                          att_W.permute(0, 2, 1)).view(batch_size, num_heads, channels // num_heads, width,
                                                       height).permute(0, 2, 1, 3, 4)

        out = self.gamma * (out_H + out_W).contiguous().view(batch_size, channels, width, height)
        out = self.out_conv(out)

        return out + x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(input_dim + input_dim, input_dim, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.size()

        # Reshape inputs to (batch_size, num_heads, width * height, channels_per_head)
        inputs = inputs.view(batch_size, channels, width * height).permute(0, 2, 1)

        # Calculate query, key, and value
        query = self.query_linear(inputs)
        key = self.key_linear(inputs)
        value = self.value_linear(inputs)

        # Reshape query, key, and value to (batch_size * num_heads, width * height, channels_per_head)
        query = query.view(batch_size * self.num_heads, width * height, self.head_dim)
        key = key.view(batch_size * self.num_heads, width * height, self.head_dim)
        value = value.view(batch_size * self.num_heads, width * height, self.head_dim)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(1, 2))
        scores = scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to value
        weighted_values = torch.matmul(attention_weights, value)

        # Reshape weighted_values to (batch_size, width * height, num_heads, channels_per_head)
        weighted_values = weighted_values.view(batch_size, width * height, self.num_heads, self.head_dim)

        # Reshape and linear transformation
        weighted_values = weighted_values.permute(0, 2, 3, 1).contiguous()
        weighted_values = weighted_values.view(batch_size, self.num_heads * self.head_dim, width, height)

        # outputs = self.output_linear(weighted_values)
        outputs = weighted_values
        return outputs

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, multi_heads=0):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        if multi_heads == 0:
            self.cca = CrissCrossAttention(inter_channels)
        else:
            self.cca = MultiHeadAttention(inter_channels, multi_heads)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class MCFF(nn.Module):
    # Mutual Control Feature Fusion.

    def __init__(self, in_channels, out_channels):
        super(MCFF, self).__init__()

        self.structure_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SENet(out_channels),
            nn.Sigmoid()
        )
        self.texture_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SENet(out_channels),
            nn.Sigmoid()
        )
        self.fusion_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SENet(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.LeakyReLU(0.2)
        )

        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.structure_beta = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))
        self.texture_beta = nn.Parameter(torch.zeros(1))

    def forward(self, structure_feature, texture_feature):
        mc_cat = torch.cat((structure_feature, texture_feature), dim=1)

        map_texture = self.structure_branch(mc_cat)
        map_structure = self.texture_branch(mc_cat)
        fusion_feature = self.fusion_branch(mc_cat)

        texture_feature_branch = texture_feature + self.texture_beta * (structure_feature * (self.texture_gamma * (map_texture * fusion_feature)))
        structure_feature_branch = structure_feature + self.structure_beta * (texture_feature * (self.structure_gamma * (map_structure * fusion_feature)))
        # structure_feature_branch = structure_feature + self.structure_gamma * (map_structure * texture_feature)

        out = self.out(torch.cat((structure_feature_branch, texture_feature_branch), dim=1))
        return out


class GCFF(nn.Module):
    # Global Control Feature Fusion.

    def __init__(self, in_channels, out_channels):
        super(GCFF, self).__init__()

        self.weight_matric = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            SENet(out_channels),
            nn.Sigmoid()
        )

        self.matric_gamma = nn.Parameter(torch.zeros(1))
        self.matric_beta = nn.Parameter(torch.zeros(1))

    def forward(self, local_feature, global_feature):
        mc_cat = torch.cat((local_feature, global_feature), dim=1)

        weight_matric = self.weight_matric(mc_cat)

        local_feature = local_feature + self.matric_beta * (global_feature * (self.matric_gamma * (weight_matric * local_feature)))

        return local_feature
