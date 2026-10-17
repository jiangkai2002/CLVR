import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 数据
x = [1, 2, 3, 4, 5]
recall = [34.05, 34.49, 35.19, 34.23, 34.22]
ndcg = [32.07, 32.41, 33.26, 32.13, 31.89]

fig, ax1 = plt.subplots(figsize=(5, 3))

# 关键：设置背景透明
fig.patch.set_alpha(0.0)  # 整个图表背景透明
ax1.patch.set_alpha(0.0)  # 左轴绘图区域背景透明
ax2 = ax1.twinx()
ax2.patch.set_alpha(0.0)  # 右轴绘图区域背景透明（避免遮挡）

# 左 y 轴 (Recall)
ax1.set_xlabel("Number of Virtual Relations", fontsize=11)
ax1.set_ylabel("Recall@20", color='tab:blue', fontsize=11)
l1, = ax1.plot(x, recall, marker='o', color='tab:blue', label="Recall", linewidth=1.8)
ax1.tick_params(axis='y', labelcolor='tab:blue', direction='inout', length=5)

# 设置左 y 轴范围
ax1.set_ylim(min(recall) - 0.5, max(recall) + 0.3)
ax1.yaxis.set_major_locator(MultipleLocator(0.3))

# 右 y 轴 (NDCG)
ax2.set_ylabel("NDCG@20", color='tab:red', fontsize=11)
l2, = ax2.plot(x, ndcg, marker='s', color='tab:red', label="NDCG", linewidth=1.8)
ax2.tick_params(axis='y', labelcolor='tab:red', direction='inout', length=5, pad=5)

# 设置右 y 轴范围
ax2.set_ylim(min(ndcg) - 0.3, max(ndcg) + 0.3)
ax2.yaxis.set_major_locator(MultipleLocator(0.3))

for label in ax2.get_yticklabels():
    label.set_position((1.02, 0))  # 稍微往右偏移一点

# 图例
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", frameon=False, fontsize=10)

# X 轴设置
ax1.set_xticks(x)
ax1.set_xticklabels(['1', '2', '3', '4', '|R|'])
ax1.tick_params(axis='x', labelsize=10)

# 关键：去掉网格（网格是背景辅助元素，需关闭）
# ax1.grid(True, linestyle='--', alpha=0.5)  # 注释掉网格代码

plt.tight_layout()
plt.show()