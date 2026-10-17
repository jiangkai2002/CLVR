import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.VRKG_old import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import matplotlib.pyplot as plt

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(
        negative_sampling(entity_pairs, train_user_set)
    ).to(device)
    return feed_dict


# =========================
# 新增：Train Recall@20
# =========================
def calc_train_recall(model, train_user_set, K=20):
    model.eval()
    with torch.no_grad():
        entity_emb, user_emb, _ = model.generate()

    hit, total = 0, 0
    for u, pos_items in train_user_set.items():
        if len(pos_items) == 0:
            continue

        scores = torch.matmul(user_emb[u], entity_emb.t())
        _, topk = torch.topk(scores, K)

        hit += len(set(topk.cpu().numpy()) & set(pos_items))
        total += len(pos_items)

    return hit / total if total > 0 else 0.0


if __name__ == '__main__':

    # 固定随机种子
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 参数 & 设备
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # 数据
    train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    train_cf_pairs = torch.LongTensor(
        np.array([[cf[0], cf[1]] for cf in train_cf], np.int32)
    )

    # 模型 & 优化器
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    best_epoch = 0
    should_stop = False

    # ===== 新增：记录曲线 =====
    epoch_list = []
    train_recall_list = []
    test_recall_list = []

    print("start training ...")

    for epoch in range(args.epoch):

        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        model.train()
        s = 0
        mf_loss_total = 0
        train_cf_s = time()

        while s + args.batch_size <= len(train_cf):
            cf_batch = get_feed_dict(
                train_cf_pairs,
                s, s + args.batch_size,
                user_dict['train_user_set']
            )

            batch_loss, mf_loss, emb_loss, contrastive_loss, attn_weights = model(cf_batch)

            optimizer.zero_grad()
            mf_loss.backward()
            optimizer.step()

            mf_loss_total += mf_loss.item()
            s += args.batch_size

        train_cf_e = time()

        if epoch % 10 == 9 or epoch == 1:

            # ========= Test =========
            test_s_t = time()
            model.eval()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            # ========= 新增 Train Recall =========
            train_recall = calc_train_recall(
                model,
                user_dict['train_user_set'],
                K=20
            )

            # ===== 记录曲线 =====
            epoch_list.append(epoch)
            train_recall_list.append(train_recall)
            test_recall_list.append(ret['recall'][3])

            # ===== 原表格 + 新增列 =====
            train_res = PrettyTable()
            train_res.field_names = [
                "Epoch", "TrainTime", "TestTime",
                "Recall(Test)", "Recall@20(Train)", "Recall@20(Test)",
                "NDCG", "Precision", "HitRatio", "AUC", "F1"
            ]
            train_res.add_row([
                epoch,
                train_cf_e - train_cf_s,
                test_e_t - test_s_t,
                ret['recall'],
                train_recall,
                ret['recall'][3],
                ret['ndcg'],
                ret['precision'],
                ret['hit_ratio'],
                ret['auc'],
                ret['f1']
            ])
            print(train_res)

            # early stopping（保持不变）
            cur_best_pre_0, stopping_step, should_stop, best_epoch = early_stopping(
                ret['recall'][3],
                cur_best_pre_0,
                stopping_step,
                best_epoch,
                epoch,
                expected_order='acc',
                flag_step=10
            )
            if should_stop:
                break

        else:
            print(
                'Epoch %d | Time %.2f | Training Loss %.4f'
                % (epoch, train_cf_e - train_cf_s, mf_loss_total)
            )

    print(f'Best Epoch: {best_epoch}, Best Recall@20(Test): {cur_best_pre_0:.4f}')

    # =========================
    # 新增：将数据保存到文件
    # =========================
    data_to_save = np.column_stack((epoch_list, train_recall_list, test_recall_list))
    np.savetxt('recall_data.csv', data_to_save, delimiter=',', header='Epoch,Train_Recall@20,Test_Recall@20', comments='')

    # =========================
    # 画 Train / Test 双曲线
    # =========================
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, train_recall_list,
             marker='s', linestyle='--', label='Train Recall@20')
    plt.plot(epoch_list, test_recall_list,
             marker='o', linestyle='-', label='Test Recall@20')

    plt.xlabel('Epoch')
    plt.ylabel('Recall@20')
    plt.title('Train vs Test Recall (Overfitting Check)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()