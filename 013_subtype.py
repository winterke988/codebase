#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

# === 设置全局字体为Arial并确保AI可编辑 ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
plt.rcParams['font.size'] = 12  # 增大基础字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # X轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度标签大小
plt.rcParams['pdf.fonttype'] = 42  # 确保在AI中可编辑
plt.rcParams['ps.fonttype'] = 42   # 确保在AI中可编辑
plt.rcParams['svg.fonttype'] = 'none'  # 确保SVG中的文本可编辑

# === 1. 读取数据 ===
data = pd.read_csv(r"E:\100-科研\667_总数据版本\010亚型对比图\LOG_0804_with_all_labels.csv")
cluster_col = 'cluster'

# === 2. 定义变量和系统映射 ===
features = [
    'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP', 'P_F', 'PH',
    'Base_Excess', 'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
    'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
    'CK-MB', 'Creatinine', 'BUN', 'Alanine_aminotransferase', 'Bilirubin',
    'Albumin', 'Aspartate_aminotransferase'
]

system_mapping = {
    'Creatinine': 'Renal', 'BUN': 'Renal',
    'Albumin': 'Hepatic', 'Bilirubin': 'Hepatic', 
    'Alanine_aminotransferase': 'Hepatic', 
    'Aspartate_aminotransferase': 'Hepatic',
    
    'Hemoglobin': 'Hematologic', 'Platelet': 'Hematologic', 
    'White_blood_cell': 'Hematologic', 'Red_blood_cell': 'Hematologic',

    'PT': 'Coagulation', 'APTT': 'Coagulation', 
    'INR': 'Coagulation', 'D_dimer': 'Coagulation',

    'Procalcitonin': 'Inflammation', 
    'Neutrophil': 'Inflammation', 'C_reactive_protein': 'Inflammation',
    'Temperature': 'Inflammation',

    'Lactate': 'Metabolic', 'Base_Excess': 'Metabolic', 'PH': 'Metabolic',

    'P_F': 'Respiratory', 'Respiratory_Rate': 'Respiratory',

    'MAP': 'Circulation', 'Heart_Rate': 'Circulation',
    'CK-MB': 'Circulation',
}

# === 3. 系统颜色定义 ===
system_colors = {
    'Inflammation': '#D95F02',     # 红橘，活跃炎症感
    'Hepatic': '#7570B3',          # 紫蓝，肝脏专业感
    'Hematologic': '#1B9E77',      # 深绿，血液稳定感
    'Coagulation': '#E7298A',      # 明亮玫红，突出凝血
    'Renal': '#66A61E',            # 青绿，肾脏功能色
    'Metabolic': '#E6AB02',        # 金黄代谢感
    'Respiratory': '#A6761D',      # 咖金褐，肺部色系
    'Circulation': '#666666'       # 中灰心血管通用色
}

# === 4. 标准化变量（Z-score） ===
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# === 5. 计算各亚型的平均值 ===
cluster_means = data.groupby(cluster_col)[features].mean().T

# === 6. 变量排序（可选：按第一个亚型） ===
sorted_vars = cluster_means[1].sort_values(ascending=False).index.tolist()

# === 7. 绘图 ===
# 增加图形尺寸以适应更大字体
fig, axes = plt.subplots(2, 2, figsize=(22, 18), sharey=False)
axes = axes.flatten()

cluster_ids = [1, 2, 3, 4]
roman_map = {1: "I", 2: "II", 3: "III", 4: "IV"}

for idx, cluster_id in enumerate(cluster_ids):
    sorted_vars = cluster_means[cluster_id].sort_values(ascending=False).index.tolist()
    plot_df = cluster_means[[cluster_id]].loc[sorted_vars].reset_index()
    plot_df.columns = ['Variable', 'Zscore']
    
    plot_df['System'] = plot_df['Variable'].map(system_mapping)
    plot_df['Color'] = plot_df['System'].map(system_colors)

    # 创建条形图
    sns.barplot(
        data=plot_df,
        y='Variable',
        x='Zscore',
        palette=plot_df['Color'],
        ax=axes[idx]
    )
    
    # 设置标题和标签（使用Arial字体）
    axes[idx].set_title(f'Subphenotype {roman_map[cluster_id]}', fontsize=16, fontname='Arial', fontweight='bold')
    axes[idx].axvline(0, color='black', linestyle='--', linewidth=1.5)
    axes[idx].set_xlabel('Z-score', fontname='Arial', fontsize=14)
    
    if idx % 2 == 0:
        axes[idx].set_ylabel('Clinical Variable', fontname='Arial', fontsize=14)
    else:
        axes[idx].set_ylabel('')
    
    # 设置刻度标签字体
    for label in axes[idx].get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(12)
    
    for label in axes[idx].get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(12)
    
    # 设置轴线的粗细
    axes[idx].spines['top'].set_linewidth(1.5)
    axes[idx].spines['right'].set_linewidth(1.5)
    axes[idx].spines['bottom'].set_linewidth(1.5)
    axes[idx].spines['left'].set_linewidth(1.5)

# === 8. 图例 ===
legend_elements = [Patch(facecolor=color, label=system, edgecolor='black', linewidth=0.5) 
                   for system, color in system_colors.items()]

# 创建图例并设置字体
legend = fig.legend(
    handles=legend_elements,
    bbox_to_anchor=(0.5, 1.0),  # ★ 修改为 (0.5, 1.0) 表示顶部中心 ★
    loc='lower center',          # ★ 修改为 'lower center' ★
    ncol=8,                      # 分成8列
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    title='Physiological System',
    fontsize=14,
    title_fontsize=14
)
# 设置图例标题和标签的字体
legend.get_title().set_fontname('Arial')
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontname('Arial')

# 调整布局以容纳图例
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部图例留出空间

# === 9. 导出为矢量图 ===
# 保存为PDF
plt.savefig(
    r"E:\100-科研\667_总数据版本\014写作和结果汇总\02图表0811单个\subphenotype_comparison2.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300
)



# 显示图形
plt.show()


# In[15]:


# 保存为SVG
plt.savefig(
    r"E:\100-科研\667_总数据版本\014写作和结果汇总\02图表0811单个\subphenotype_comparison2.svg",
    format="svg",
    bbox_inches="tight",
    dpi=600
)


# In[21]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from matplotlib.colors import to_rgb
import os

# === 设置全局字体为 Arial（保证AI里可修改文字） ===
plt.rcParams['font.family'] = 'Arial'

# === 1. 数据读取与标准化 ===
data = pd.read_csv(r"E:\100-科研\667_总数据版本\010亚型对比图\LOG_0804_with_all_labels.csv")

features = [
    'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP', 'P_F', 'PH',
    'Base_Excess', 'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
    'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
    'CK-MB', 'Creatinine', 'BUN', 'Alanine_aminotransferase', 'Bilirubin',
    'Albumin', 'Aspartate_aminotransferase'
]
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# === 2. 系统分类及颜色 ===
system_mapping = {
    'Creatinine': 'Renal', 'BUN': 'Renal',
    'Albumin': 'Hepatic', 'Bilirubin': 'Hepatic',
    'Alanine_aminotransferase': 'Hepatic',
    'Aspartate_aminotransferase': 'Hepatic',
    'Hemoglobin': 'Hematologic', 'Platelet': 'Hematologic',
    'White_blood_cell': 'Hematologic', 'Red_blood_cell': 'Hematologic',
    'PT': 'Coagulation', 'APTT': 'Coagulation',
    'INR': 'Coagulation', 'D_dimer': 'Coagulation',
    'Procalcitonin': 'Inflammation', 'Neutrophil': 'Inflammation',
    'C_reactive_protein': 'Inflammation', 'Temperature': 'Inflammation',
    'Lactate': 'Metabolic', 'Base_Excess': 'Metabolic', 'PH': 'Metabolic',
    'P_F': 'Respiratory', 'Respiratory_Rate': 'Respiratory',
    'MAP': 'Circulation', 'Heart_Rate': 'Circulation',
    'CK-MB': 'Circulation',
}
system_colors = {
    'Inflammation': '#D95F02',
    'Hepatic': '#7570B3',
    'Hematologic': '#1B9E77',
    'Coagulation': '#E7298A',
    'Renal': '#66A61E',
    'Metabolic': '#E6AB02',
    'Respiratory': '#A6761D',
    'Circulation': '#666666'
}

# === 3. 聚类方法、颜色、映射（Agglomerative 放最前） ===
cluster_methods = {
    'cluster': 'Agglomerative',
    'GMM_cluster': 'GMM',
    'Class': 'LPA'
}
line_styles = {
    'Agglomerative': 'solid',
    'GMM': 'dashed',
    'LPA': 'dotted'
}
roman_map = {1: "I", 2: "II", 3: "III", 4: "IV"}
base_colors = {
    1: '#79A3D9',  # 蓝
    2: '#7B967A',  # 灰绿
    3: '#F9C77E',  # 黄橙
    4: '#CE4257'   # 红
}

# === 4. 亮度渐变函数 ===
def get_shaded_color(base_color, shade_level, total):
    r, g, b = to_rgb(base_color)
    factor = 1.2 - (shade_level / (total - 1)) * 0.6
    return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0))

# === 5. 两两亚型组合 ===
all_subtypes = [1, 2, 3, 4]
pair_list = list(combinations(all_subtypes, 2))

# === 6. 输出目录 ===
output_dir = r"E:\100-科研\667_总数据版本\014写作和结果汇总\02图表0811单个"
os.makedirs(output_dir, exist_ok=True)

# === 7. 主循环绘图（无图例） ===
for sub1, sub2 in pair_list:
    plt.figure(figsize=(18, 6))

    # 排序变量（用 GMM）
    ref_means = data.groupby("GMM_cluster")[features].mean().T
    diff = ref_means[sub1] - ref_means[sub2]
    sorted_vars = diff.sort_values(ascending=False).index.tolist()
    
    for idx, (method_col, method_name) in enumerate(cluster_methods.items()):
        means = data.groupby(method_col)[features].mean().T
        if sub1 in means.columns and sub2 in means.columns:
            shade1 = get_shaded_color(base_colors[sub1], idx, len(cluster_methods))
            shade2 = get_shaded_color(base_colors[sub2], idx, len(cluster_methods))

            plt.plot(sorted_vars, means[sub1][sorted_vars],
                     color=shade1,
                     linestyle=line_styles[method_name],
                     linewidth=2,
                     marker='o')

            plt.plot(sorted_vars, means[sub2][sorted_vars],
                     color=shade2,
                     linestyle=line_styles[method_name],
                     linewidth=2,
                     marker='o')

    # x轴变量分类着色
    ax = plt.gca()
    label_colors = [system_colors[system_mapping[var]] for var in sorted_vars]
    for tick, color in zip(ax.get_xticklabels(), label_colors):
        tick.set_color(color)

    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Z-score", fontsize=12)
    plt.title(f"{roman_map[sub1]} vs {roman_map[sub2]} ", fontsize=14)
    plt.tight_layout()

    # 保存PDF（无图例，AI里可改文字）
    filename = f"Subtype_{roman_map[sub1]}_vs_{roman_map[sub2]}.pdf"
    plt.savefig(os.path.join(output_dir, filename), format='pdf', dpi=300)
    plt.close()

print(f"✅ 主图已保存到：{output_dir}")

# === 8. 单独生成图例（Agglomerative 优先显示） ===
plt.figure(figsize=(8, 4))
y_pos = 0
height = 0.5

for subtype in [1, 2, 3, 4]:
    base_color = base_colors[subtype]
    for i, (method_col, method_name) in enumerate(cluster_methods.items()):
        color = get_shaded_color(base_color, i, len(cluster_methods))
        plt.hlines(y=y_pos, xmin=0, xmax=2, colors=[color],
                   linestyles=line_styles[method_name], linewidth=4)
        label = f"{roman_map[subtype]} - {method_name}"
        plt.text(2.1, y_pos, label, va='center', fontsize=12, family='Arial')
        y_pos -= height
    y_pos -= height/2  # 亚型间空隙

plt.xlim(0, 4)
plt.ylim(y_pos + height, 1)
plt.axis('off')
#plt.title("Subtype and Clustering Method Legend", fontsize=14, family='Arial')
plt.tight_layout()

legend_file = os.path.join(output_dir, "Subtype_Legend.pdf")
plt.savefig(legend_file, format='pdf', dpi=300)
plt.close()

print(f"✅ 单独图例已保存到：{legend_file}")


# In[23]:


# === 9. 系统分类颜色图例 ===
plt.figure(figsize=(6, 4))
y_pos = 0
height = 0.5

for system, color in system_colors.items():
    plt.hlines(y=y_pos, xmin=0, xmax=2, colors=[color], linewidth=6)
    plt.text(2.3, y_pos, system, va='center', fontsize=12, family='Arial')
    y_pos -= height

plt.xlim(0, 4)
plt.ylim(y_pos + height, 1)
plt.axis('off')

# === 这里加标题 ===
plt.title("Category", fontsize=14, family='Arial')

plt.tight_layout()

system_legend_file = os.path.join(output_dir, "System_Legend.pdf")
plt.savefig(system_legend_file, format='pdf', dpi=300)
plt.close()

print(f"✅ 系统颜色图例已保存到：{system_legend_file}")



# In[ ]:




