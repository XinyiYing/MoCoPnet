from math import sqrt
# import matplotlib.pyplot as plt
import functools
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
# from thop import profile

class Net(nn.Module): 
    def __init__(self, upscale_factor, num_frame=7, nf=64):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_conv = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1)
        self.deep_conv = RDG_cdc(G0=nf, C=6, G=32, n_RDB=4)
        self.att = AttModule(nf)
        self.fusion = Fusion(num_frame, nf)
        self.reconstruct = RDG(G0=nf, C=6, G=32, n_RDB=8)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * upscale_factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        b, n, c0, h, w = x.shape            # B, N, 1, H, W
        ref_bicubic = F.interpolate(x[:, n // 2, :, :, :], scale_factor=self.upscale_factor, mode='bicubic',
                                 align_corners=False)           # B, 1, aH, aW

        buffer = self.init_conv(x.contiguous().view(-1, c0, h, w))  # B*N, C, H, W
        buffer = self.deep_conv(buffer)                             # B*N, C, H, W
        buffer = self.att(buffer.contiguous().view(b, n, -1, h, w))     # B, G, 3C, H, W
        buffer = self.fusion(buffer)                            # B, C, H, W
        buffer = self.reconstruct(buffer)                       # B, C, H, W
        out = self.upscale(buffer) + ref_bicubic
        return out


class AttModule(nn.Module):
    def __init__(self, nf):
        super(AttModule, self).__init__()
        self.align1 = AttAlign(kernel=3, dilation=3, rd=8, nf=nf)
        self.align2 = AttAlign(kernel=3, dilation=1, rd=8, nf=nf)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x_ref = x[:, n // 2, :, :, :]
        out = []
        for i in range(n//2):
            x_n1, x_n2 = x[:, i, :, :, :], x[:, -1-i, :, :, :]
            x_n1, x_n2 = self.align1(x_n1, x_ref), self.align1(x_n2, x_ref)
            x_n1, x_n2 = self.align2(x_n1, x_ref), self.align2(x_n2, x_ref)
            out.append(torch.cat((x_n1, x_ref, x_n2), dim=1))
        out = torch.stack(out, dim=1)       # B * G * 3C * H * W

        return out


class AttAlign(nn.Module):
    def __init__(self, kernel, dilation, rd, nf):
        super(AttAlign, self).__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.conv_q = nn.Conv2d(nf, nf // rd, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_k = nn.Conv2d(nf, nf // rd, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, ref):
        b, c, h, w = x.shape # x B * C * H * W
        buffer = self.conv_k(x) # x B * C/rd * H * W
        K, V = [], []
        for i in range(self.kernel): # [0,1,2]
            for j in range(self.kernel): # [0,1,2]
                u, v = (i - self.kernel//2) * self.dilation, (j - self.kernel//2) * self.dilation # [-1,0,1], [-1,0,1]
                delta_w = 2/(w-1)*u
                delta_h = 2/(h-1)*v
                grid = np.meshgrid(range(w), range(h))
                grid = np.stack(grid, axis=-1).astype(np.float64)
                grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1 + delta_w
                grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1 + delta_h
                grid = grid.transpose(2, 0, 1)
                grid = np.tile(grid, (b, 1, 1, 1))
                grid = Variable(torch.Tensor(grid))
                if x.is_cuda ==  True:
                    grid = grid.to(x.device)
                grid = grid.transpose(1, 2)
                grid = grid.transpose(3, 2)
                K.append(F.grid_sample(buffer, grid=grid, mode='bilinear',align_corners=True))
                V.append(F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True))
        K = torch.stack(K, dim=1)             # x B * 9 * C/rd * H * W
        V = torch.stack(V, dim=1)             # x B * 9 * C * H * W
        bufferQ = self.conv_q(ref)            # ref B * C/rd * H * W
        Q = bufferQ.unsqueeze(1).repeat(1, self.kernel**2, 1, 1, 1)      # ref B * 9 * C/rd * H * W
        score = torch.sum(K * Q, dim=2)                     # B * 9 * H * W
        attmap = nn.Softmax(1)(score)                       # B * 9 * H * W
        out = V * attmap.unsqueeze(2).repeat(1, 1, c, 1, 1)  # x [B * 9 * C * H * W] * att  C * [B* 9 * H * W]
        out = torch.sum(out, dim=1)                          # B * C * H * W

        return out


class Fusion(nn.Module):
    def __init__(self, num_frame, nf):
        super(Fusion, self).__init__()
        num_inter = (num_frame - 1) // 2
        self.rdb_intra = RDB(G0=3 * nf, C=4, G=64)
        self.conv_intra = nn.Conv2d(3 * nf, nf, kernel_size=1, stride=1, padding=0)
        self.rdb_inter = RDB(G0=(num_inter * nf), C=4, G=64)
        self.conv_inter = nn.Conv2d(num_inter * nf, nf, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, g, c, h, w = x.size()        # B, G, 3C, H, W
        temp = []
        for i in range(g):
            buffer = x[:, i, :, :, :]
            temp.append(self.conv_intra(self.rdb_intra(buffer)))
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv_inter(self.rdb_inter(buffer_cat))
        return out


class OneConv(nn.Module):
    def __init__(self, G0, G):
        super(OneConv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(OneConv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out



class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            kernel_diff2 = kernel_diff1[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class OneConv_cdc(nn.Module):
    def __init__(self, G0, G):
        super(OneConv_cdc, self).__init__()
        self.conv = Conv2d_cd(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)

class RDB_cdc(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB_cdc, self).__init__()
        convs = []
        for i in range(C):
            if i == 0:
                convs.append(OneConv_cdc(G0+i*G, G))
            else:
                convs.append(OneConv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG_cdc(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG_cdc, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            # if i ==0:
            #     RDBs.append(RDB_cdc(G0, C, G))
            # else:
            #     RDBs.append(RDB(G0, C, G))
            RDBs.append(RDB_cdc(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out

if __name__ == "__main__":
    net = Net(4).cuda()
    input = torch.randn(1, 7, 1, 64, 64).cuda()
    # net = AttModule(64)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


