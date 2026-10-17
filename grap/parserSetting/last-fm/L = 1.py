import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
iteration_q = [1, 2, 3, 4]
recall = [36.85, 36.69, 36.69, 36.69]
ndcg = [23.05, 23.00, 23.00, 23.00]

# 设置更美观的字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 创建图形和双坐标轴
fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')

# 绘制 Recall 曲线（左轴）
line1 = ax1.plot(
    iteration_q, recall,
    marker='o', color='#E74C3C',  # 柔和红色
    label='Recall', linewidth=2,
    markersize=8, markerfacecolor='white', markeredgewidth=2  # 空心标记+边框
)
ax1.set_xlabel('Iteration Q', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall@20', color='#E74C3C', fontsize=12, fontweight='bold')
ax1.set_ylim(36, 37)
ax1.grid(axis='y', linestyle='--', alpha=0.7, color='#EEEEEE')  # 添加网格

# 设置横轴只显示整数刻度
ax1.set_xticks(iteration_q)

# 关键设置：刻度线向内显示，不伸出轴外
ax1.tick_params(
    axis='both',          # 同时设置x轴和y轴
    direction='in',       # 刻度线向内
    labelsize=10,
    color='#666666',      # 刻度线颜色
    width=1.2,            # 刻度线宽度
    labelcolor='black'    # 刻度标签颜色
)
ax1.tick_params(axis='y', labelcolor='#E74C3C')  # 单独设置y轴标签颜色

# 创建右坐标轴并绘制 NDCG 曲线
ax2 = ax1.twinx()
line2 = ax2.plot(
    iteration_q, ndcg,
    marker='o', color='#F1C40F',  # 柔和黄色
    label='NDCG', linewidth=2,
    markersize=8, markerfacecolor='white', markeredgewidth=2
)
ax2.set_ylabel('NDCG@20', color='#F1C40F', fontsize=12, fontweight='bold')
ax2.set_ylim(22, 23.1)

# 右轴同样设置刻度线向内
ax2.tick_params(
    axis='y',             # 只设置y轴
    direction='in',       # 刻度线向内
    labelsize=10,
    color='#666666',
    width=1.2,
    labelcolor='#F1C40F'  # 右轴标签颜色
)

# 添加标题和图例（统一管理图例）
ax1.set_title('$L=1$', fontsize=14, fontweight='bold', pad=20)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.85, 0.85),
           frameon=True, framealpha=0.9, fontsize=10, shadow=False)

# 优化坐标轴边框样式
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#666666')
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#666666')

# 调整布局并显示
plt.tight_layout()
plt.show()