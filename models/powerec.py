# -*- coding: utf-8 -*-


import numpy as np
import scipy.sparse as sp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class POWERec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(POWERec, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.num_modal = 3
        self.emb_size = self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.neg_weight = config['neg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.prompt_num = config['prompt_num']  # float32 type: the weight decay for l2 normalizaton
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # define layers and loss
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim)))

        # normalized adj matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        self.id_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.latent_dim)))
        self.v_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.latent_dim)))
        self.t_prompt = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.prompt_num, self.latent_dim)))

        self.id_model = LayerGCN(self.n_users, self.n_items, self.user_embeddings, self.item_embeddings, self.latent_dim, self.id_prompt)
        self.v_model = LayerGCN(self.n_users, self.n_items, self.user_embeddings, self.v_feat, self.latent_dim, self.v_prompt)
        self.t_model = LayerGCN(self.n_users, self.n_items, self.user_embeddings, self.t_feat, self.latent_dim, self.t_prompt)

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()

        self.mf_loss = BPRLoss()
        self.reg_loss = L2Loss()

    # def post_epoch_processing(self):
    #     with torch.no_grad():
    #         return '=== Layer weights: {}'.format(F.softmax(self.layer_weights.exp(), dim=0))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len)).to(self.device)
        else:
            # pruning edges by pro
            keep_idx = torch.multinomial(self.edge_values, keep_len)         # prune high-degree nodes
        self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], 0)
        return ego_embeddings

    def forward(self, adj):
        user_id, item_id = self.id_model(adj)
        user_v, item_v = self.v_model(adj)
        user_t, item_t = self.t_model(adj)

        u_embeddings = torch.cat([user_id, user_v, user_t], 1)
        i_embeddings = torch.cat([item_id, item_v, item_t], 1)

        return u_embeddings, i_embeddings

    def bpr_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        u_embeddings = u_embeddings[user]
        posi_embeddings = i_embeddings[pos_item]
        negi_embeddings = i_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))

        weak_modality, modality_indicator = self.find_weak_modality(u_embeddings, posi_embeddings, negi_embeddings)
        fake_neg_pos_e = (1 - weak_modality) * posi_embeddings
        fake_neg_neg_e = weak_modality * negi_embeddings
        fake_neg_e = fake_neg_pos_e + fake_neg_neg_e  ## [bzs, num_model * dim]
        fake_neg_scores = torch.mul(u_embeddings, fake_neg_e).sum(1)
        bpr_loss += self.neg_weight * torch.sum(-m(pos_scores - fake_neg_scores))

        return bpr_loss

    def emb_loss(self, user, pos_item, neg_item):
        # calculate BPR Loss
        u_ego_embeddings = self.user_embeddings[user]
        posi_ego_embeddings = self.item_embeddings[pos_item]
        negi_ego_embeddings = self.item_embeddings[neg_item]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        return reg_loss

    def find_weak_modality(self, user_e, pos_e, neg_e):
        # user_e = F.normalize(user_e, p=2, dim=1)
        # pos_e = F.normalize(pos_e, p=2, dim=1)
        # neg_e = F.normalize(neg_e, p=2, dim=1)
        pos_score_ = torch.mul(user_e, pos_e).view(-1, self.num_modal, self.emb_size).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, self.num_modal, self.emb_size).sum(dim=-1)
        modality_indicator = (pos_score_ - neg_score_).softmax(-1).detach()

        weak_modality = (modality_indicator == modality_indicator.min(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)
        weak_modality = torch.tile(weak_modality.view(-1, self.num_modal, 1), [1, 1, self.emb_size])
        weak_modality = weak_modality.view(-1, self.num_modal * self.emb_size)
        return weak_modality, modality_indicator

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward(self.masked_adj)

        mf_loss = self.bpr_loss(user_all_embeddings, item_all_embeddings, user, pos_item, neg_item)
        reg_loss = self.emb_loss(user, pos_item, neg_item)

        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj_matrix)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

class LayerGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, user_fea, item_fea, emb_size, prompt_embedding):
        super(LayerGCN, self).__init__()
        self.n_layers = 4
        self.num_user = num_user
        self.num_item = num_item
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.emb_size = emb_size
        self.prompt_embedding = prompt_embedding
        self.mlp = nn.Sequential(nn.Linear(self.item_fea.shape[1], self.emb_size), nn.Tanh())

    def forward(self, adj):
        '''
        user_fea: [num_user, emb_size]
        item_fea: [num_user, fea_size]
        '''
        prompt_embd = torch.sum(self.prompt_embedding, 0)  ## [emb]
        user_embd = self.user_fea + prompt_embd[None, :]  ## [user, emb_size]
        item_embd = self.mlp(self.item_fea)  ## [item_num, emb_size]

        ego_embeddings = torch.cat((user_embd, item_embd), dim=0)
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        u_embd, i_embd = torch.split(ui_all_embeddings, [self.num_user, self.num_item])

        return u_embd, i_embd
