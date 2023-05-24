# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
LATTICE
################################################
Reference:
    https://github.com/CRIPAC-DIG/LATTICE
    ACM MM'2021: [Mining Latent Structures for Multimedia Recommendation] 
    https://arxiv.org/abs/2104.09036
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class LATTICE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LATTICE, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size'] # 64
        self.feat_embed_dim = config['feat_embed_dim'] # 64
        self.weight_size = config['weight_size'] # [64, 64]
        self.knn_k = config['knn_k'] # 10
        self.lambda_coeff = config['lambda_coeff'] # 0.9
        self.cf_model = config['cf_model'] # lightgcn
        self.n_layers = config['n_layers'] # 1
        self.reg_weight = config['reg_weight'] # [0.0, 1e-05, 1e-04, 1e-03]中取0.0
        self.build_item_graph = True

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32) # user-item交互矩阵 shape (19445, 7050)
        self.norm_adj = self.get_adj_mat() # 获取邻接矩阵，并且对其归一化
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device) # 将scipy的稀疏矩阵转为torch的稀疏张量
        self.item_adj = None

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size # [64,64,64]
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim) # 嵌入层
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim) # 嵌入层
        nn.init.xavier_uniform_(self.user_embedding.weight) # 使张量的元素均匀分布？U(-a,a) shape torch.Size([19445, 64])
        nn.init.xavier_uniform_(self.item_id_embedding.weight) # shape torch.Size([7050, 64])

        if config['cf_model'] == 'ngcf': # 下游cf使用ngcf
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config['mess_dropout']
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k)) # image_adj_10.pt
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k)) # text_adj_10.pt

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False) # v_feat是visual特征, shape是torch.Size([7050, 4096])
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file) # 加载图邻接矩阵，shape:torch.Size([7050, 7050])
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian(image_adj)
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda() if config['use_gpu'] else image_adj.cpu() # 图原始邻接矩阵shape:torch.Size([7050, 7050])

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file) # 加载文字邻接矩阵，shape:torch.Size([7050, 7050])
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() if config['use_gpu'] else text_adj.cpu() # 文字原始邻接矩阵shape:torch.Size([7050, 7050])

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim) # in_feat 4096 out_feat 64
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim) # in_feat 384 out_feat 64

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def pre_epoch_processing(self):
        self.build_item_graph = True

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32) # shape (26495, 26495) 19445+7050
        adj_mat = adj_mat.tolil() # 返回稀疏矩阵的lil_matrix形式 lil_matrix:基于行连接存储的稀疏矩阵(Row-based linked list sparse matrix)
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok() # 返回稀疏矩阵的dok_matrix形式

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            #print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo() # 返回稀疏矩阵的coo_matrix形式 coo_matrix:坐标格式的矩阵(Coodrdinate format matrix)

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0])) # 邻接矩阵归一化
        return norm_adj_mat.tocsr() # 返回稀疏矩阵的csr_matrix形式 csr_matrix: 压缩稀疏行矩阵(Compressed sparse row matrix)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, build_item_graph=False):
        # 公式(4) ？感觉像？ 因为下方调用是线性回归
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            # 2.2
            if self.v_feat is not None:
                self.image_adj = build_sim(image_feats) # 公式(1)
                self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k) # 公式(2)
                learned_adj = self.image_adj
                original_adj = self.image_original_adj
            if self.t_feat is not None:
                self.text_adj = build_sim(text_feats) # 公式(1)
                self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k) # 公式(2)
                learned_adj = self.text_adj
                original_adj = self.text_original_adj
            if self.v_feat is not None and self.t_feat is not None:
                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj # 公式(6)
                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj # 公式(6)

            learned_adj = compute_normalized_laplacian(learned_adj) # 公式(3) ？有个疑问，为什么没有对original_adj进行归一化？
            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj # 公式(5)，learned_adj表示邻接矩阵A_波浪_m，original_adj表示模特感知图结构S_波浪_m
        else:
            self.item_adj = self.item_adj.detach()

        # 2.3 图卷积
        h = self.item_id_embedding.weight # 第一层的h
        for i in range(self.n_layers): # 遍历所有层
            h = torch.mm(self.item_adj, h) # 公式(7)，更新第l层的h

        # 2.4 结合CF
        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'lightgcn': # 下游cf是lightgcn
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

    def bpr_loss(self, users, pos_items, neg_items): #  https://zhuanlan.zhihu.com/p/620570517
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1) # torch.mul矩阵点乘
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores) # 公式(10)中的σ 激活函数
        mf_loss = -torch.mean(maxi) # 求均值，添负号 这里使用的是这个公式https://blog.csdn.net/qq_35541614/article/details/103816504

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users] # shape torch.Size([2048, 64])
        pos_i_g_embeddings = ia_embeddings[pos_items] # shape torch.Size([2048, 64])
        neg_i_g_embeddings = ia_embeddings[neg_items] # shape torch.Size([2048, 64])

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings) # 公式(10)
        return batch_mf_loss + batch_emb_loss + batch_reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, build_item_graph=True)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

