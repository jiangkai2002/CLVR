import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt


class Aggregator(nn.Module):
    """
    Local Weighted Smoothing aggregation scheme with Relation-aware Attention
    """

    def __init__(self, n_users, n_virtual, n_iter, emb_size):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.n_iter = n_iter
        self.emb_size = emb_size

        self.projection_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )
        self.tau = 0.1

        # 关系注意力权重计算
        self.relation_attn = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),  # 输入是[entity_emb, relation_emb]拼接
            nn.ReLU(),
            nn.Linear(emb_size, 1)  # 输出每个关系的注意力分数
        )

    def forward(self, entity_emb, user_emb, edge_index, edge_type, adj_mat):
        device = entity_emb.device
        n_entities = entity_emb.shape[0]
        edge_type_uni = torch.unique(edge_type)
        relation_emb_list = []  # 存储不同关系类型的聚合结果

        """Step 1: 对每种关系类型分别聚合"""
        for i in edge_type_uni:
            index = torch.where(edge_type == i)[0]
            head, tail = edge_index[0][index], edge_index[1][index]

            # 迭代聚合
            u = None
            neigh_emb = entity_emb[tail]
            for clus_iter in range(self.n_iter):
                if u is None:
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)
                else:
                    center_emb = u[head]
                    sim = torch.sum(center_emb * neigh_emb, dim=1)
                    neigh_emb = sim.unsqueeze(1) * neigh_emb  # 加权邻居
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)

                if clus_iter < self.n_iter - 1:
                    squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                    u = squash.unsqueeze(1) * F.normalize(u, dim=1)
                u += entity_emb  # 残差连接

            relation_emb_list.append(u)  # 保存当前关系类型的聚合结果

        """Step 2: 关系注意力加权聚合"""
        # 将不同关系的结果堆叠 [n_relations, n_entities, emb_size]
        all_relation_embs = torch.stack(relation_emb_list, dim=0)

        # 计算注意力分数
        # 输入: entity_emb和relation_emb拼接 [n_entities, n_relations, 2*emb_size]
        entity_emb_expanded = entity_emb.unsqueeze(1).expand(-1, len(edge_type_uni), -1)
        attn_input = torch.cat([entity_emb_expanded, all_relation_embs.permute(1, 0, 2)], dim=-1)

        # 计算注意力权重 [n_entities, n_relations]
        attn_scores = self.relation_attn(attn_input).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)  # 按关系维度归一化

        # 加权求和 [n_entities, emb_size]
        entity_agg = torch.sum(attn_weights.unsqueeze(-1) * all_relation_embs.permute(1, 0, 2), dim=1)

        """Step 3: 用户聚合"""
        user_index, item_index = adj_mat.nonzero()
        user_index = torch.tensor(user_index, dtype=torch.long).to(device)
        item_index = torch.tensor(item_index, dtype=torch.long).to(device)

        u = None
        for clus_iter in range(self.n_iter):
            neigh_emb = entity_emb[item_index]
            if u is None:
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=self.n_users, dim=0)
            else:
                center_emb = u[user_index]
                sim = torch.sum(center_emb * neigh_emb, dim=1)
                neigh_emb = sim.unsqueeze(1) * neigh_emb
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=self.n_users, dim=0)

            if clus_iter < self.n_iter - 1:
                squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                u = squash.unsqueeze(1) * F.normalize(u, dim=1)
            u += user_emb

        user_agg = u

        contrastive_loss = self.calculate_relation_contrastive_loss(entity_agg, edge_index, edge_type)

        # 返回注意力权重
        return entity_agg, user_agg, contrastive_loss, attn_weights

    def calculate_relation_contrastive_loss(self, entity_emb, edge_index, edge_type):
        """
        关系感知的对比学习（带分批处理）
        """
        # 获取头节点和尾节点表示
        head_emb = entity_emb[edge_index[0]]
        tail_emb = entity_emb[edge_index[1]]

        # 通过投影头
        head_proj = self.projection_head(head_emb)
        tail_proj = self.projection_head(tail_emb)

        # 归一化
        head_proj = F.normalize(head_proj, dim=1)
        tail_proj = F.normalize(tail_proj, dim=1)

        # 按关系类型分组
        unique_relations = torch.unique(edge_type)
        contrastive_loss = 0
        total_pairs = 0  # 用于加权平均

        # 设置batch大小（根据你的GPU内存调整）
        batch_size = 256  # 可以调整为512/2048等
        for rel in unique_relations:
            # 获取当前关系的边索引
            mask = (edge_type == rel)
            rel_head = head_proj[mask]
            rel_tail = tail_proj[mask]

            n_pairs = rel_head.size(0)
            if n_pairs == 0:
                continue

            # 分批处理
            batch_loss = 0
            for i in range(0, n_pairs, batch_size):
                # 获取当前batch
                batch_head = rel_head[i:i + batch_size]
                batch_tail = rel_tail[i:i + batch_size]

                # 计算相似度矩阵（只计算对角线附近区域节省内存）
                with torch.no_grad():
                    # 计算每个head与对应tail的相似度作为正样本
                    pos_sim = (batch_head * batch_tail).sum(dim=1, keepdim=True) / self.tau

                    # 随机采样负样本（避免计算全矩阵）
                    neg_head = batch_head[torch.randperm(batch_head.size(0))]
                    neg_tail = batch_tail[torch.randperm(batch_tail.size(0))]

                    # 计算负样本相似度
                    neg_sim_head = torch.mm(batch_head, neg_tail.t()) / self.tau
                    neg_sim_tail = torch.mm(neg_head, batch_tail.t()) / self.tau

                # 构建logits
                logits_head = torch.cat([pos_sim, neg_sim_head], dim=1)
                logits_tail = torch.cat([pos_sim, neg_sim_tail.t()], dim=1)

                # 对比损失
                labels = torch.zeros(batch_head.size(0), dtype=torch.long).to(entity_emb.device)
                loss_head = F.cross_entropy(logits_head, labels)
                loss_tail = F.cross_entropy(logits_tail, labels)

                batch_loss += (loss_head + loss_tail) / 2 * (min(batch_size, n_pairs - i) / n_pairs)

            contrastive_loss += batch_loss
            total_pairs += 1

        if total_pairs > 0:
            contrastive_loss = contrastive_loss / total_pairs
        else:
            contrastive_loss = torch.tensor(0.0).to(entity_emb.device)

        return contrastive_loss


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_iter, n_users,
                 n_virtual, n_relations, adj_mat, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.convs = nn.ModuleList()
        self.adj_mat = adj_mat
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2
        self.tau = 0.1  # 温度参数
        self.contrastive_loss_weight = 0.1

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(
                Aggregator(n_users=n_users, n_virtual=n_virtual, n_iter=n_iter, emb_size=channel)  # 新增 emb_size
            )
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                adj_mat, interact_mat, mess_dropout=True, node_dropout=False):

        # 原始视图
        entity_emb_orig = entity_emb
        user_emb_orig = user_emb

        # 生成扰动视图
        if node_dropout:
            edge_index_view1, edge_type_view1 = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            edge_index_view2, edge_type_view2 = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
        else:
            edge_index_view1, edge_type_view1 = edge_index, edge_type
            edge_index_view2 = edge_index.clone()
            edge_type_view2 = edge_type.clone()

        # 视图1
        entity_res_emb1 = entity_emb_orig
        user_res_emb1 = user_emb_orig
        cor1 = 0

        # 视图2
        entity_res_emb2 = entity_emb_orig
        user_res_emb2 = user_emb_orig
        cor2 = 0

        weight = self.weight
        relation_ = torch.mm(weight, latent_emb.t())
        relation_remap = torch.argmax(relation_, dim=1)
        edge_type_view1 = relation_remap[edge_type_view1 - 1]
        edge_type_view2 = relation_remap[edge_type_view2 - 1]

        total_contrastive_loss = 0  # 用于累积所有层的对比损失
        all_attn_weights = []  # 存储所有层的注意力权重

        for i in range(len(self.convs)):
            # 视图1 - 现在接收四个返回值
            entity_emb1, user_emb1, contrastive_loss1, attn_weights1 = self.convs[i](
                entity_emb_orig, user_emb_orig,
                edge_index_view1, edge_type_view1, adj_mat
            )
            all_attn_weights.append(attn_weights1)

            # 视图2 - 现在接收四个返回值
            entity_emb2, user_emb2, contrastive_loss2, attn_weights2 = self.convs[i](
                entity_emb_orig, user_emb_orig,
                edge_index_view2, edge_type_view2, adj_mat
            )

            # 累积对比损失
            total_contrastive_loss += (contrastive_loss1 + contrastive_loss2) / 2

            if mess_dropout:
                entity_emb1 = self.dropout(entity_emb1)
                user_emb1 = self.dropout(user_emb1)
                entity_emb2 = self.dropout(entity_emb2)
                user_emb2 = self.dropout(user_emb2)

            entity_emb1 = F.normalize(entity_emb1)
            user_emb1 = F.normalize(user_emb1)
            entity_emb2 = F.normalize(entity_emb2)
            user_emb2 = F.normalize(user_emb2)

            entity_res_emb1 = torch.add(entity_res_emb1, entity_emb1)
            user_res_emb1 = torch.add(user_res_emb1, user_emb1)
            entity_res_emb2 = torch.add(entity_res_emb2, entity_emb2)
            user_res_emb2 = torch.add(user_res_emb2, user_emb2)

        # 计算图级别的对比损失
        graph_contrastive_loss = self.calculate_graph_contrastive_loss(
            torch.cat([user_res_emb1, entity_res_emb1], dim=0),
            torch.cat([user_res_emb2, entity_res_emb2], dim=0)
        )

        # 总对比损失 = 各层对比损失平均 + 图级别对比损失
        total_contrastive_loss = total_contrastive_loss / len(self.convs) + graph_contrastive_loss

        # 返回原始视图的结果、总对比损失和最后一层的注意力权重
        return entity_res_emb1, user_res_emb1, cor1 + self.contrastive_loss_weight * total_contrastive_loss, \
        all_attn_weights[-1]

    def calculate_graph_contrastive_loss(self, z1, z2):
        """
        计算图结构对比损失
        z1, z2: 两个不同视图下的节点表示
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / self.tau)

        # 正样本是对角线元素
        pos_sim = torch.diag(sim_matrix)

        # 负样本是其他所有元素
        neg_sim = torch.sum(sim_matrix, dim=1) - pos_sim

        # InfoNCE损失
        loss = -torch.log(pos_sim / neg_sim).mean()

        return loss


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.kg_l2loss_lambda = args_config.kg_l2loss_lambda
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_iter = args_config.n_iter
        self.n_virtual = args_config.n_virtual  # The number of virtual relation types
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.gcn = self._init_model()

        self.tau = args_config.tau if hasattr(args_config, 'tau') else 0.1  # 温度参数
        self.contrastive_loss_weight = args_config.contrastive_loss_weight if hasattr(args_config,
                                                                                      'contrastive_loss_weight') else 0.1

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_virtual, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_iter=self.n_iter,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_virtual=self.n_virtual,
                         adj_mat=self.adj_mat,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, cf_batch):
        user = cf_batch['users']
        pos_item = cf_batch['pos_items']
        neg_item = cf_batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor, attn_weights = self.gcn(user_emb,
                                                                   item_emb,
                                                                   self.latent_emb,
                                                                   self.edge_index,
                                                                   self.edge_type,
                                                                   self.adj_mat,
                                                                   self.interact_mat,
                                                                   mess_dropout=self.mess_dropout,
                                                                   node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        # 计算原始BPR损失
        bpr_loss, mf_loss, emb_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # 计算对比学习损失
        contrastive_loss = self.calculate_contrastive_loss(user_gcn_emb, entity_gcn_emb, user, pos_item)

        # 总损失 = BPR损失 + 对比学习损失
        total_loss = bpr_loss + self.contrastive_loss_weight * contrastive_loss

        return total_loss, mf_loss, emb_loss, contrastive_loss, attn_weights

    def calculate_contrastive_loss(self, user_emb, item_emb, users, pos_items):
        """
        计算用户-物品对比学习损失
        """
        # 获取batch中的用户和正样本物品表示
        batch_user_emb = user_emb[users]  # [batch_size, emb_size]
        batch_pos_item_emb = item_emb[pos_items]  # [batch_size, emb_size]
        # 归一化
        batch_user_emb = F.normalize(batch_user_emb, dim=1)
        batch_pos_item_emb = F.normalize(batch_pos_item_emb, dim=1)
        # 计算相似度矩阵
        logits = torch.mm(batch_user_emb, batch_pos_item_emb.t()) / self.tau  # [batch_size, batch_size]
        # 对比学习目标 - 对角线元素为正样本对
        labels = torch.arange(logits.size(0)).to(self.device)
        # 对称的对比损失
        loss_u = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.t(), labels)
        contrastive_loss = (loss_u + loss_i) / 2
        return contrastive_loss

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.adj_mat,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss + emb_loss, emb_loss

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

    def visualize_attention(self, user_ids=None, item_ids=None, save_path=None):
        """
        可视化注意力权重
        :param user_ids: 要可视化的用户ID列表，如果为None则随机选择3个
        :param item_ids: 要可视化的物品ID列表，如果为None则随机选择3个
        :param save_path: 图片保存路径，如果为None则直接显示
        """
        # 获取嵌入和注意力权重
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        _, _, _, attn_weights = self.gcn(
            user_emb, item_emb, self.latent_emb,
            self.edge_index, self.edge_type,
            self.adj_mat, self.interact_mat,
            mess_dropout=False, node_dropout=False
        )

        # 将注意力权重转换为numpy数组
        attn_weights = attn_weights.detach().cpu().numpy()

        # 处理用户ID
        if user_ids is None:
            user_ids = np.random.choice(self.n_users, size=min(3, self.n_users), replace=False)
        elif isinstance(user_ids, int):
            user_ids = [user_ids]

        # 处理物品ID
        if item_ids is None:
            item_ids = np.random.choice(self.n_items, size=min(3, self.n_items), replace=False)
        elif isinstance(item_ids, int):
            item_ids = [item_ids]

        plt.figure(figsize=(15, 10))

        # 可视化用户注意力
        for i, user_id in enumerate(user_ids):
            plt.subplot(2, len(user_ids) + len(item_ids), i + 1)
            user_attn = attn_weights[user_id]
            plt.bar(range(len(user_attn)), user_attn)
            plt.title(f"User {user_id} Attention")
            plt.xlabel("Relation Type")
            plt.ylabel("Attention Weight")

        # 可视化物品注意力
        for j, item_id in enumerate(item_ids):
            plt.subplot(2, len(user_ids) + len(item_ids), len(user_ids) + j + 1)
            item_attn = attn_weights[self.n_users + item_id]
            plt.bar(range(len(item_attn)), item_attn)
            plt.title(f"Item {item_id} Attention")
            plt.xlabel("Relation Type")
            plt.ylabel("Attention Weight")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()