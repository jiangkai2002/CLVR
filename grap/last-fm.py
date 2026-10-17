import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 数据
x = [1, 2, 3, 4, 5]
recall = [37.83, 38.79, 39.51, 38.28, 38.19]
ndcg = [24.49, 24.91, 25.63, 24.80, 24.69]

# 绘图
fig, ax1 = plt.subplots(figsize=(5, 3))

# 左 y 轴 (Recall)
ax1.set_xlabel("Number of Virtual Relations")
ax1.set_ylabel("Recall@20", color='tab:blue')
l1, = ax1.plot(x, recall, marker='o', color='tab:blue', label="Recall")
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 设置左 y 轴范围（稍微留一点空白）
ax1.set_ylim(min(recall) - 0.5, max(recall) + 0.5)
ax1.yaxis.set_major_locator(MultipleLocator(0.5))  # 设置刻度间距

# 右 y 轴 (NDCG)
ax2 = ax1.twinx()
ax2.set_ylabel("NDCG@20", color='tab:red')
l2, = ax2.plot(x, ndcg, marker='o', color='tab:red', label="NDCG")
ax2.tick_params(axis='y', labelcolor='tab:red')

# 设置右 y 轴范围（与左轴比例相似）
ax2.set_ylim(min(ndcg) - 0.3, max(ndcg) + 0.3)
ax2.yaxis.set_major_locator(MultipleLocator(0.3))  # 设置刻度间距

# 图例合并
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="center")

# X轴刻度
ax1.set_xticks(x)
ax1.set_xticklabels(['1', '2', '3', '4', '|R|'])

plt.tight_layout()
plt.show()
