import torch
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from util.config import config as cfg
from util.misc import fill_hole, regularize_sin_cos
from nmslib import lanms
from util.pbox import bbox_transfor_inv
from layers.utils import EuclideanDistances, normalize_adj


class Graph_RPN(object):
    def __init__(self, pooling, pst_dim):
        self.pst_dim = pst_dim
        self.NodePooling = pooling
        self.k_at_hop = cfg.k_at_hop
        self.active_connection = cfg.active_connection
        self.depth = len(self.k_at_hop)

        self.tr_thresh = cfg.tr_thresh
        self.tcl_thresh = cfg.tcl_thresh
        self.expend = cfg.expend + 1.0
        self.clip = (4, 8)

    @staticmethod
    def PositionalEncoding(geo_map, model_dim):
        shape = geo_map.shape
        model_dim = model_dim // shape[1]

        # t0 = time.time()
        pp = np.array([np.power(1000, 2.0 * (j // 2) / model_dim)
                       for j in range(model_dim)]).reshape(model_dim, 1, 1)
        ps = np.repeat(np.expand_dims(geo_map, axis=0), model_dim, axis=0)
        pst_encoding = ps / pp
        # print("A time: {}".format(time.time() - t0))
        pst_encoding[:, 0::2] = np.sin(pst_encoding[:, 0::2])
        pst_encoding[:, 1::2] = np.cos(pst_encoding[:, 1::2])
        pst_encoding = pst_encoding.reshape((shape[0], -1))

        return pst_encoding

    def graph_generate(self, knn_graph, feat_map):

        """
        return the vertex feature and the adjacent matrix A, together
        with the indices of the center node and its 1-hop nodes
        :param knn_graph:
        :param feat:
        :param labels:
        :return:
        """
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        knn_graph = knn_graph[:, :self.k_at_hop[0] + 1]
        hops_list = list()
        one_hops_list = list()
        for index, cluster in enumerate(knn_graph):
            hops = list()
            center_idx = index

            hops.append(set(knn_graph[center_idx][1:]))
            one_hops_list.append(set(knn_graph[center_idx][1:]))
            # Actually we dont need the loop since the depth is fixed here,
            # But we still remain the code for further revision
            for d in range(1, self.depth):
                hops.append(set())
                for h in hops[-2]:
                    hops[-1].update(set(knn_graph[h][1:self.k_at_hop[d] + 1]))

            hops_set = set([h for hop in hops for h in hop])
            nodes_list = list(hops_set)
            nodes_list.insert(0, center_idx)

            hops_list.append(nodes_list)

        max_num_nodes = max([len(ips) for ips in hops_list])

        feat_batch = list()
        adj_batch = list()
        h1id_batch = list()
        cid_batch = list()
        unique_ips_batch = list()

        for idx, ips in enumerate(hops_list):
            num_nodes = int(len(ips))
            center_idx = ips[0]
            one_hops = one_hops_list[idx]
            unique_nodes_map = {j: i for i, j in enumerate(ips)}

            center_node = torch.tensor([unique_nodes_map[center_idx], ]).type(torch.long)
            one_hop_idcs = torch.tensor([unique_nodes_map[i] for i in one_hops], dtype=torch.long)
            center_feat = feat_map[torch.tensor(center_idx, dtype=torch.long)]
            feat = feat_map[torch.tensor(ips, dtype=torch.long)] - center_feat

            A = np.zeros((num_nodes, num_nodes))
            feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, feat.shape[1]).cuda()], dim=0)

            for node in ips:
                neighbors = knn_graph[node, 1:self.active_connection + 1]
                for n in neighbors:
                    if n in ips:
                        A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                        A[unique_nodes_map[n], unique_nodes_map[node]] = 1

            A = normalize_adj(A, type="DAD")
            A_ = torch.zeros(max_num_nodes, max_num_nodes)
            A_[:num_nodes, :num_nodes] = A

            # Testing
            unique_ips = torch.tensor(ips)
            unique_ips = torch.cat([unique_ips, torch.zeros(max_num_nodes - num_nodes, dtype=torch.long)], dim=0)

            feat_batch.append(feat)
            adj_batch.append(A_)
            cid_batch.append(center_node)
            h1id_batch.append(one_hop_idcs)
            unique_ips_batch.append(unique_ips)

        feat_bth = torch.stack(feat_batch, 0)
        adj_bth = torch.stack(adj_batch, 0)
        cid_bth = torch.stack(cid_batch, 0)
        h1id_bth = torch.stack(h1id_batch, 0)
        unique_ips_bth = torch.stack(unique_ips_batch, 0)

        return feat_bth, adj_bth, cid_bth, h1id_bth, unique_ips_bth

    def proposals_layer(self, tr_map, tcl_map, radii_map, sin_map, cos_map):

        tr_pred_mask = tr_map > self.tr_thresh
        tcl_pred_mask = tcl_map > self.tcl_thresh

        # multiply TR and TCL
        tcl_mask = tcl_pred_mask * tr_pred_mask

        # regularize
        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)

        # find disjoint regions
        tcl_mask = fill_hole(tcl_mask)
        tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(tcl_mask)
        proposals = None
        for cont in tcl_contours:
            deal_map = mask.copy()
            cv2.drawContours(deal_map, [cont], -1, 1, -1)
            if deal_map.sum() <= 100:
                continue
            text_map = tr_map * deal_map
            # ## 1. Reverse generation of box
            bboxs = bbox_transfor_inv(radii_map, sin_map, cos_map, text_map, wclip=self.clip, expend=self.expend)

            # ## 3. local nms
            bboxs = lanms.merge_quadrangle_n9(bboxs.astype('float32'), 0.25)

            reconstruct_mask = mask.copy()
            boxes = bboxs[:, :8].reshape((-1, 4, 2)).astype(np.int32)

            cv2.drawContours(reconstruct_mask, boxes, -1, 1, -1)
            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                continue

            if proposals is None:
                proposals = bboxs
            else:
                proposals = np.concatenate([proposals, bboxs], axis=0)

        if proposals is None or proposals.shape[0] <= 0:
            return None, None

        # ## 5. generate cluster label
        cxy = np.mean(proposals[:, :8].reshape((-1, 4, 2)), axis=1).astype(np.int32)

        # ## 6. Geometric features
        gh = (radii_map[:, :, 0] + radii_map[:, :, 1])
        h_map = gh[cxy[:, 1], cxy[:, 0]]
        w_map = np.clip(h_map // 2, 2*self.clip[0], 2*self.clip[1])
        c_map = cos_map[cxy[:, 1], cxy[:, 0]]
        s_map = sin_map[cxy[:, 1], cxy[:, 0]]
        geo_map = np.stack([cxy[:, 0], cxy[:, 1], h_map, w_map, c_map, s_map], axis=1)

        return geo_map, proposals

    def __call__(self, image, output, graph_feat):

        image = image[0].data.cpu().numpy()
        tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred = output[0, 4].data.cpu().numpy()
        cos_pred = output[0, 5].data.cpu().numpy()
        radii_pred = output[0, 6:8].permute(1, 2, 0).contiguous().data.cpu().numpy()

        out_result = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred
        }

        roi_map, proposals = self.proposals_layer(tr_pred[1], tcl_pred[1], radii_pred, sin_pred, cos_pred)

        if roi_map is None:
            return True, (0, 0, 0, 0, 0, 0, out_result)

        # ## 1. compute euclidean similarity
        similarity_e = np.array(EuclideanDistances(roi_map[:, 0:2], roi_map[:, 0:2])) / cos_pred.shape[0]

        # ## 2. position embedding
        pos_feat = self.PositionalEncoding(roi_map, self.pst_dim)
        pos_feat = torch.from_numpy(pos_feat).cuda().float()

        # ## 3. generate graph node feature
        batch_id = np.zeros((roi_map.shape[0], 1), dtype=np.float32)
        roi_map = np.hstack((batch_id, roi_map.astype(np.float32, copy=False)))
        roi_map = torch.from_numpy(roi_map).cuda()

        roi_feat = self.NodePooling(graph_feat, roi_map)
        roi_feat = roi_feat.view(roi_feat.shape[0], -1)
        node_feat = torch.cat((roi_feat, pos_feat), dim=-1)

        # # ## 4. computing cosine similarity of Node feature
        # roi_feature_np = node_feat.data.cpu().numpy()
        # similarity_c = 1.0001 - cosine_similarity(roi_feature_np)
        # similarity_matrix = (similarity_e + similarity_c) / 2.0
        similarity_matrix = similarity_e

        # ## 5. compute the knn graph
        knn_graph = np.argsort(similarity_matrix, axis=1)[:, :]
        feat_bth, adj_bth, cid_bth, h1id_bth, unique_ips_bth = self.graph_generate(knn_graph, node_feat)

        return False, (feat_bth, adj_bth, cid_bth, h1id_bth, unique_ips_bth, proposals, out_result)
