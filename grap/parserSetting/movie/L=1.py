import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
iteration_q = [1, 2, 3, 4]
recall = [34.23, 34.43, 34.44, 34.44]
ndcg = [32.13, 32.35, 32.33, 32.33]

# 创建图形和双坐标轴
fig, ax1 = plt.subplots(figsize=(8, 6))

# 绘制 Recall 曲线（左轴）
ax1.plot(iteration_q, recall, marker='o', color='red', label='Recall', linewidth=2)
ax1.set_xlabel('Iteration Q')
ax1.set_ylabel('Recall@20', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(34.20, 34.50)

# 设置横轴只显示整数刻度
ax1.set_xticks(iteration_q)  # 直接使用数据中的整数作为刻度

# 创建右坐标轴并绘制 NDCG 曲线
ax2 = ax1.twinx()
ax2.plot(iteration_q, ndcg, marker='o', color='gold', label='NDCG', linewidth=2)
ax2.set_ylabel('NDCG@20', color='gold')
ax2.tick_params(axis='y', labelcolor='gold')
ax2.set_ylim(32.10, 32.40)

# 添加标题和图例
ax1.set_title('$L=1$')
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

plt.tight_layout()
plt.show()