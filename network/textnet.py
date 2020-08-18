import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GCN
from layers import KnnGraph
from RoIlayer import RROIAlign
from layers import Graph_RPN
from network.vgg import VggNet
from network.resnet import ResNet


class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.class_channel = 6
        self.reg_channel = 2

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=True)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=True)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 32)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=True)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=True)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 256)
            self.merge3 = UpBlok(512 + 256, 128)
            self.merge2 = UpBlok(256 + 128, 64)
            self.merge1 = UpBlok(64 + 64, 32)
        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)

        return up1, up2, up3, up4, up5


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.k_at_hop = [8, 4]
        self.post_dim = 120
        self.active_connection = 3
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, self.is_training)
        self.gcn_model = GCN(600, 32)  # 600 = 480 + 120
        self.pooling = RROIAlign((3, 4), 1.0 / 1)  # (32+8)*3*4 =480

        # ##class and regression branch
        self.out_channel = 8
        self.predict = nn.Sequential(
            #nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, self.out_channel, kernel_size=1, stride=1, padding=0)
        )

        # ## gcn branch
        if is_training:
            self.graph = KnnGraph(self.k_at_hop, self.active_connection, self.pooling, 120, self.is_training)
        else:
            self.graph = Graph_RPN(self.pooling, 120)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x, roi_data=None, to_device=None):
        up1, up2, up3, up4, up5 = self.fpn(x)
        predict_out = self.predict(up1)

        graph_feat = torch.cat([up1, predict_out], dim=1)
        feat_batch, adj_batch, h1id_batch, gtmat_batch = self.graph(graph_feat, roi_data)
        gcn_pred = self.gcn_model(feat_batch, adj_batch, h1id_batch)

        return predict_out, (gcn_pred, to_device(gtmat_batch))

    def forward_test(self, img):
        up1, up2, up3, up4, up5 = self.fpn(img)
        predict_out = self.predict(up1)

        return predict_out

    def forward_test_graph(self, img):
        up1, up2, up3, up4, up5 = self.fpn(img)
        predict_out = self.predict(up1)

        graph_feat = torch.cat([up1, predict_out], dim=1)

        flag, datas = self.graph(img, predict_out, graph_feat)
        feat, adj, cid, h1id, node_list, proposals, output = datas
        if flag:

            return None, None, None, output

        adj, cid, h1id = map(lambda x: x.cuda(), (adj, cid, h1id))
        gcn_pred = self.gcn_model(feat, adj, h1id)

        pred = F.softmax(gcn_pred, dim=1)

        edges = list()
        scores = list()
        node_list = node_list.long().squeeze().cpu().numpy()
        bs = feat.size(0)

        for b in range(bs):
            cidb = cid[b].int().item()
            nl = node_list[b]
            for j, n in enumerate(h1id[b]):
                n = n.item()
                edges.append([nl[cidb], nl[n]])
                scores.append(pred[b * (h1id.shape[1]) + j, 1].item())

        edges = np.asarray(edges)
        scores = np.asarray(scores)

        return edges, scores, proposals, output
