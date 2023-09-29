import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
from torch.autograd import Variable

from .netvlad import NetVladWrapper
from .transformer import TransformerBlock, DiffTransformerBlock
from .pooling import GeM, MAC, SPoC




class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_sparse = self.avg_pool(x)

        # from IPython import embed;embed()

        # Apply 1D convolution along the channel dimension
        y = self.conv(y_sparse.F.unsqueeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).squeeze(-1)
        # y is (batch_size, channels) tensor

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor

        y_sparse = ME.SparseTensor(y, coordinate_manager=y_sparse._manager,
                                   coordinate_map_key=y_sparse.coordinate_map_key)
        # y must be features reduced to the origin
        return self.broadcast_mul(x, y_sparse)


class GatedInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, keep_dim=3):
        super().__init__()
        self.keep_dim = keep_dim
        assert keep_dim <= 5 and keep_dim > 0
        # receptive field 1x1
        if in_channels != out_channels:
            self.conv1x1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dimension=3),
                ME.MinkowskiBatchNorm(
                    out_channels)
            )
        else:
            self.conv1x1 = lambda x: x

        # receptive field 3x3
        self.conv3x3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels)
        )

        # receptive field 5x5
        self.conv5x5 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels)
        )

        # receptive field 7x7
        self.conv7x7 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
        )

        # receptive field 9*9
        self.conv9x9 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
        )

        self.convs = [self.conv1x1, self.conv3x3, self.conv5x5,
                      self.conv7x7, self.conv9x9]

        self.trans_layers = nn.ModuleList()

        for i in range(len(self.convs)):
            self.trans_layers.append(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        out_channels, out_channels, kernel_size=1, stride=1, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(
                        out_channels, out_channels, kernel_size=1, stride=1, dimension=3),
                    ME.MinkowskiBatchNorm(out_channels)
                )
            )
        self.trans_layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                out_channels, len(self.convs)*out_channels, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(len(self.convs)*out_channels),
        )

        self.eca1x1 = ECALayer(out_channels)
        self.eca3x3 = ECALayer(out_channels)
        self.eca5x5 = ECALayer(out_channels)
        self.eca7x7 = ECALayer(out_channels)
        self.eca9x9 = ECALayer(out_channels)

    def wrap(self, x, F):
        return ME.SparseTensor(
            F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x._manager,
        )

    def forward(self, x):
        x0 = self.conv1x1(x)
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        x4 = self.conv9x9(x)

        # from IPython import embed;embed()

        x0 = self.eca1x1(x0)
        x1 = self.eca3x3(x1)
        x2 = self.eca5x5(x2)
        x3 = self.eca7x7(x3)
        x4 = self.eca9x9(x4)

        w0 = self.trans_layers[0](x0)
        w1 = self.trans_layers[1](x1)
        w2 = self.trans_layers[2](x2)
        w3 = self.trans_layers[3](x3)
        w4 = self.trans_layers[4](x4)

        w = w0+w1+w2+w3+w4
        w = self.trans_layer2(w)

        f = w.F.reshape(w.F.shape[0], 5, -1)
        f = F.softmax(f, dim=1).permute(1, 0, 2)

        if self.keep_dim != 5:
            # idx = torch.argsort(f, dim=0)[:3]
            _, idx = torch.topk(f, self.keep_dim, dim=0)
            mask = torch.zeros(f.shape).to(idx.device)
            mask.scatter_(0, idx, 1.)
            f = f * mask

        w0, w1, w2, w3, w4 = self.wrap(w, f[0]), self.wrap(
            w, f[1]), self.wrap(w, f[2]), self.wrap(w, f[3]), self.wrap(w, f[4])
        x = w0*x0+w1*x1+w2*x2+w3*x3+w4*x4

        return x

class STN3d(nn.Module):
    def __init__(self, cfg):
        super(STN3d, self).__init__()

        self.cfg = cfg
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.conv.append(nn.Conv1d(3, cfg.conv_channels[0], 1))
        self.bn.append(nn.BatchNorm1d(cfg.conv_channels[0]))

        for i in range(len(cfg.conv_channels) - 1):
            self.conv.append(nn.Conv1d(cfg.conv_channels[i], cfg.conv_channels[i + 1], 1))
            self.bn.append(nn.BatchNorm1d(cfg.conv_channels[i + 1]))


        self.fc.append(nn.Linear(cfg.conv_channels[-1], cfg.fc_channels[0]))
        self.bn.append(nn.BatchNorm1d(cfg.fc_channels[0]))

        for i in range(len(cfg.fc_channels) - 1):
            self.fc.append(nn.Linear(cfg.fc_channels[i], cfg.fc_channels[i + 1]))
            self.bn.append(nn.BatchNorm1d(cfg.fc_channels[i + 1]))

        self.fc.append(nn.Linear(cfg.fc_channels[-1], 9))

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2,1)

        for i in range(len(self.conv)):
            x = F.relu(self.bn[i](self.conv[i](x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.cfg.conv_channels[-1])

        for i in range(len(self.fc) - 1):
            x = F.relu(self.bn[i + len(self.conv)](self.fc[i](x)))

        x = self.fc[-1](x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    # def __init__(self, global_feat = True, feature_transform = False):
    def __init__(self, cfg):
        super(PointNetfeat, self).__init__()

        self.cfg = cfg
        self.stn = STN3d(cfg.std_cfg)

        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.conv.append(nn.Conv1d(3, cfg.channels[0], 1))
        self.bn.append(nn.BatchNorm1d(cfg.channels[0]))

        for i in range(len(cfg.channels) - 1):
            self.conv.append(nn.Conv1d(cfg.channels[i], cfg.channels[i + 1], 1))
            self.bn.append(nn.BatchNorm1d(cfg.channels[i + 1]))

        self.global_feat = cfg.global_feat

    def forward(self, x):
        n_pts = x.size()[1]
        trans = self.stn(x)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn[0](self.conv[0](x)))

        # if self.feature_transform:
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2,1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2,1)
        # else:
        #     trans_feat = None

        pointfeat = x

        for i in range(len(self.cfg.channels) - 2):
            x = F.relu(self.bn[i + 1](self.conv[i + 1](x)))

        x = self.bn[-1](self.conv[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.cfg.channels[-1])

        if self.global_feat:
            return x
            # return x, trans
        else:
            x = x.view(-1, self.cfg.channels[-1], 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)
            # return torch.cat([x, pointfeat], 1), trans


class Ours3DFPN(nn.Module):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(self, cfg, q_size):
        super().__init__()
        self.cfg = cfg
        self.skip_conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections
        self.q_size = q_size
        self.fine_to_coarse = cfg.fine_to_coarse

        self.up_conv0_lst = nn.ModuleList()
        self.up_convs_lst = nn.ModuleList()
        self.up_blocks_lst = nn.ModuleList()

        # The first convolution is special case, with kernel size = 5
        for i in range(len(self.q_size)):
            self.up_convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
            self.up_blocks = nn.ModuleList()   # Bottom-up blocks
            self.down_convs = nn.ModuleList()   # Top-down tranposed convolutions

            self.up_conv0 = nn.Sequential(
                ME.MinkowskiConvolution(cfg.up_conv_cfgs[i][0].in_channels, cfg.up_conv_cfgs[i][0].out_channels, kernel_size=cfg.up_conv_cfgs[i][0].kernel_size, stride=cfg.up_conv_cfgs[i][0].stride,
                                        dimension=3),
                ME.MinkowskiBatchNorm(cfg.up_conv_cfgs[i][0].out_channels),
                ME.MinkowskiReLU(inplace=True)
            )

            for idx, up_conv_cfg in enumerate(cfg.up_conv_cfgs[i][1:]):
                self.up_convs.append(
                    nn.Sequential(
                        ME.MinkowskiConvolution(up_conv_cfg.in_channels, up_conv_cfg.out_channels, kernel_size=up_conv_cfg.kernel_size, stride=up_conv_cfg.stride,
                                                dimension=3),
                        ME.MinkowskiBatchNorm(up_conv_cfg.out_channels),
                        ME.MinkowskiReLU(inplace=True)
                    )
                )
                
                self.up_blocks.append(GatedInceptionBlock(
                    up_conv_cfg.out_channels, 
                    cfg.transformer_cfg.global_channels if idx == len(cfg.up_conv_cfgs[i])-2 else cfg.up_conv_cfgs[i][idx+2].in_channels,
                    cfg.keep_channels if hasattr(cfg, 'keep_channels') else 3)
                )

            self.up_conv0_lst.append(self.up_conv0)
            self.up_convs_lst.append(self.up_convs)
            self.up_blocks_lst.append(self.up_blocks)

        self.diff_transformer = DiffTransformerBlock(self.cfg.transformer_cfg, q_size, self.cfg.step_size)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(cfg.transformer_cfg.global_channels*(len(q_size)+1),
                      self.cfg.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.cfg.out_channels),
            nn.ReLU(inplace=True)
        )

        # self.pointnet = nn.Sequential(
        #     PointNetfeat(self.cfg.pointnet_cfg),
        #     nn.Linear(self.cfg.pointnet_cfg.channels[-1], self.cfg.transformer_cfg.global_channels),
        #     nn.BatchNorm1d(self.cfg.transformer_cfg.global_channels)
        # )
        


    def forward(self, data):

        xs, pcd = data
        
        # f_p = self.pointnet(pcd)
        f_p = None


        # from IPython import embed;embed()
        new_xs = []
        if self.fine_to_coarse:
            order = list(range(len(xs)))
        else:
            order = list(range(len(xs))[::-1])

        for i in order:

            x = self.up_conv0_lst[i](xs[i])

            for _, (conv, block) in enumerate(zip(self.up_convs_lst[i], self.up_blocks_lst[i])):
                x = conv(x)
                x = block(x)

            features = x.decomposed_features

            x = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
            x = x.permute(0, 2, 1)

            new_xs.append(x)

        # x = torch.cat(new_xs, dim=2)
        # (batch_size, feature_size, n_points)
        x = self.diff_transformer(new_xs, f_p)
        x = self.conv_fuse(x)
        return x


class Ours(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Ours3DFPN(cfg.backbone_cfg, cfg.quantization_size)

        assert cfg.backbone_cfg.out_channels == cfg.pool_cfg.in_channels

        if cfg.pool_cfg.type == 'Max':
            assert cfg.pool_cfg.in_channels == cfg.pool_cfg.out_channels
            self.pool = MAC(cfg.pool_cfg)
        elif cfg.pool_cfg.type == 'Avg':
            assert cfg.pool_cfg.in_channels == cfg.pool_cfg.out_channels
            self.pool = SPoC(cfg.pool_cfg)
        elif cfg.pool_cfg.type == 'GeM':
            assert cfg.pool_cfg.in_channels == cfg.pool_cfg.out_channels
            self.pool = GeM(cfg.pool_cfg)
        elif cfg.pool_cfg.type == 'NetVlad':
            self.pool = NetVladWrapper(cfg.pool_cfg)
        else:
            raise NotImplementedError(
                'Pool type has not implemented: {}'.format(cfg.pool_cfg.type))

    def forward(self, x):
        x = self.backbone(x)
        # assert len(x.shape) == 2
        # x: [bs, feat_size]
        x = self.pool(x)
        return x
