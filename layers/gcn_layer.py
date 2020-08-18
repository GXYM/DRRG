import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from layers.utils import EuclideanDistances, normalize_adj
import time


class KnnGraph(object):
    def __init__(self, k_at_hop, active_connection, pooling, pst_dim, is_train=True):
        self.pst_dim = pst_dim
        self.NodePooling = pooling
        self.is_train = is_train
        self.k_at_hop = k_at_hop
        self.depth = len(self.k_at_hop)
        self.active_connection = active_connection
        self.cluster_threshold = 0.75

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

    def localIPS(self, knn_graph, labels_gt=None):
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        knn_graph = knn_graph[:, :self.k_at_hop[0] + 1]
        hops_list = list()
        one_hops_list = list()
        for index, cluster in enumerate(knn_graph):
            hops = list()
            center_idx = index

            h0 = set(knn_graph[center_idx][1:])
            hops.append(h0)
            # Actually we dont need the loop since the depth is fixed here,
            # But we still remain the code for further revision
            for d in range(1, self.depth):
                hops.append(set())
                for h in hops[-2]:
                    hops[-1].update(set(knn_graph[h][1:self.k_at_hop[d] + 1]))

            hops_set = set([h for hop in hops for h in hop])
            nodes_list = list(hops_set)
            nodes_list.insert(0, center_idx)

            for idx, ips in enumerate(hops_list):
                union = len(list(set(ips).union(set(nodes_list))))
                intersection = len(list(set(ips).intersection(set(nodes_list))))
                iou = intersection / (union + 1e-5)
                if iou > self.cluster_threshold \
                        and center_idx in one_hops_list[idx] \
                        and labels_gt[ips[0]] == labels_gt[center_idx]\
                        and labels_gt[ips[0]] != 0:
                    break
                    
            else:  # not break for loop , performance this code
                hops_list.append(nodes_list)
                one_hops_list.append(h0)

        return hops_list,  one_hops_list

    def graph_IPS(self, feat_bin, labels_bin, hops_bin, one_hops_bin, knn_graph_bin):

        max_num_nodes = max([len(ips) for hops in hops_bin for ips in hops])

        feat_batch = list()
        adj_batch = list()
        h1id_batch = list()
        gtmat_batch = list()
        for bind, knn_graph in enumerate(knn_graph_bin):
            feat_map = feat_bin[bind]
            hops_list = hops_bin[bind]
            one_hops_list = one_hops_bin[bind]
            labels_gt = labels_bin[bind]

            for idx, ips in enumerate(hops_list):
                num_nodes = int(len(ips))
                center_idx = ips[0]
                one_hops = one_hops_list[idx]
                unique_nodes_map = {j: i for i, j in enumerate(ips)}

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

                labels = torch.from_numpy(labels_gt[ips]).type(torch.long)
                one_hop_labels = labels[one_hop_idcs]
                edge_labels = ((labels_gt[center_idx] == one_hop_labels)
                               & labels_gt[center_idx] > 0).long()

                feat_batch.append(feat)
                adj_batch.append(A_)
                h1id_batch.append(one_hop_idcs)
                gtmat_batch.append(edge_labels)

        feat_bth = torch.stack(feat_batch, 0)
        adj_bth = torch.stack(adj_batch, 0)
        h1id_bth = torch.stack(h1id_batch, 0)
        gtmat_bth = torch.stack(gtmat_batch, 0)

        return feat_bth, adj_bth, h1id_bth, gtmat_bth

    def __call__(self, feats, gt_data=None):

        knn_graph_bin = list()
        hops_bin = list()
        one_hops_bin = list()
        feat_bin = list()
        labels_bin = list()
        gt_data = gt_data.numpy()
        for bind in range(gt_data.shape[0]):
            roi_num = int(gt_data[bind, 0, 0])
            img_size = int(gt_data[bind, 0, 8])
            geo_map = gt_data[bind, :roi_num, 1:7]
            label = gt_data[bind, :roi_num, 7].astype(np.int32)

            # ## 1. compute euclidean similarity
            ctr_xy = geo_map[:, 0:2]
            similarity_e = np.array(EuclideanDistances(ctr_xy, ctr_xy), dtype=np.float) / img_size

            # ## 2. position embedding
            pos_feat = self.PositionalEncoding(geo_map, self.pst_dim)
            pos_feat = torch.from_numpy(pos_feat).cuda().float()

            # ## 3. generate graph node feature
            batch_id = np.zeros((geo_map.shape[0], 1), dtype=np.float32) * bind
            roi_map = np.hstack((batch_id, geo_map.astype(np.float32, copy=False)))
            roi_map = torch.from_numpy(roi_map).cuda()

            roi_feat = self.NodePooling(feats[bind].unsqueeze(0), roi_map)
            roi_feat = roi_feat.view(roi_feat.shape[0], -1)
            node_feat = torch.cat((roi_feat, pos_feat), dim=-1)

            # # ## 4. computing cosine similarity of Node feature
            # roi_feature_np = node_feat.data.cpu().numpy()
            # similarity_c = 1.0001 - cosine_similarity(roi_feature_np)
            # similarity_matrix = (similarity_e + similarity_c) / 2.0
            similarity_matrix = similarity_e

            # ## 5. compute the knn graph
            knn_graph = np.argsort(similarity_matrix, axis=1)[:, :]
            hops, one_hops = self.localIPS(knn_graph, label)

            # ## 6. Packing data
            feat_bin.append(node_feat)
            labels_bin.append(label)
            hops_bin.append(hops)
            one_hops_bin.append(one_hops)
            knn_graph_bin.append(knn_graph)

        batch_data = self.graph_IPS(feat_bin, labels_bin, hops_bin, one_hops_bin, knn_graph_bin)

        return batch_data













