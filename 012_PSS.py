#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# 1. 读取数据
file_path = r"E:\100-科研\667_总数据版本\007层次聚类\插补后数据_cluster0805.csv"
df = pd.read_csv(file_path)

# 2. 连续、二分类、多分类变量
continuous_vars = [
    'Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
    'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
    'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
    'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
    'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
    'Bilirubin', 'Albumin', 'Aspartate_aminotransferase',
    'Blood_Purification_Time', 'Mv_Time', 'PSS', 'ICU_LOS', 'LOS',
    'ICU_FDS', 'Mv_FDS', 'Survival_Time'
]
binary_vars = [
    '28_day_mortality', 'Gender',
    'Blood_Purification_Modality', 'HVHF', 'Heparin', 'Citrate', 'Sedation',
    'Nutrition_Method', 'Mechanical_ventilation', 'Epinephrine',
    'Norepinephrine', 'Dopamine'
]
categorical_vars = ['cluster']  # 多分类变量

# 3. PSS 四分位分组
df['PSS_group'] = pd.qcut(df['PSS'], 4, labels=['<25%', '25%-50%', '50%-75%', '>75%'])

# 4. 构建基线表
summary = pd.DataFrame()

# 总人数
summary.loc['Patient number', 'Overall'] = len(df)
for g in df['PSS_group'].cat.categories:
    summary.loc['Patient number', g] = (df['PSS_group'] == g).sum()

# 连续变量：中位数 [IQR]
for var in continuous_vars:
    overall_median = df[var].median()
    overall_iqr = df[var].quantile([0.25, 0.75]).values
    summary.loc[var, 'Overall'] = f"{overall_median:.1f} [{overall_iqr[0]:.1f},{overall_iqr[1]:.1f}]"
    for g in df['PSS_group'].cat.categories:
        subset = df[df['PSS_group'] == g][var]
        median = subset.median()
        q1, q3 = subset.quantile([0.25, 0.75])
        summary.loc[var, g] = f"{median:.1f} [{q1:.1f},{q3:.1f}]"

# 二分类变量：n (%)
for var in binary_vars:
    overall_n = df[var].sum()
    summary.loc[var, 'Overall'] = f"{overall_n} ({overall_n/len(df)*100:.1f}%)"
    for g in df['PSS_group'].cat.categories:
        subset = df[df['PSS_group'] == g]
        n = subset[var].sum()
        summary.loc[var, g] = f"{n} ({n/len(subset)*100:.1f}%)"

# 多分类变量（如 cluster）：每个类别单独一行
for var in categorical_vars:
    for level in sorted(df[var].dropna().unique()):
        row_name = f"{var}={level}"
        overall_n = (df[var] == level).sum()
        summary.loc[row_name, 'Overall'] = f"{overall_n} ({overall_n/len(df)*100:.1f}%)"
        for g in df['PSS_group'].cat.categories:
            subset = df[df['PSS_group'] == g]
            n = (subset[var] == level).sum()
            summary.loc[row_name, g] = f"{n} ({n/len(subset)*100:.1f}%)"

# 5. 导出
#summary.to_excel(r"E:\100-科研\667_总数据版本\014PSS四分类\PSS_baseline_table.xlsx")

print("✅ 基线表已保存为 PSS_baseline_table.xlsx")


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
file_path = r"E:\100-科研\667_总数据版本\007层次聚类\插补后数据_cluster0805.csv"
df = pd.read_csv(file_path)

# 2. 你的连续变量
continuous_vars = [
    'Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
    'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
    'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
    'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
    'CK-MB', 'Creatinine', 'BUN', 'Alanine_aminotransferase',
    'Bilirubin', 'Albumin', 'Aspartate_aminotransferase'
]

# 3. 按 PSS 四分位分组（也可以改成你的 cluster 变量）
df['PSS_group'] = pd.qcut(df['PSS'], 4, labels=['Quantile1', 'Quantile2', 'Quantile3', 'Quantile4'])

# 4. 对连续变量做 Z-score 标准化
df_zscore = df[continuous_vars].apply(lambda x: (x - x.mean()) / x.std())

# 把分组加进去
df_zscore['Group'] = df['PSS_group']

# 5. 计算每组的均值
df_group_mean = df_zscore.groupby('Group').mean().reset_index()

# 6. 宽表转长表（方便 seaborn 画图）
df_long = df_group_mean.melt(id_vars='Group', var_name='Variable', value_name='Standardized mean')

# 7. 画图
plt.figure(figsize=(12, 6))
sns.barplot(data=df_long, x='Group', y='Standardized mean', hue='Variable', palette='tab20')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Variables in quantile PSS score', fontsize=14)
plt.ylabel('Standardized mean')
plt.xlabel('PSS Quantile Subset clusters')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Variables')
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ---------- 设置字体 & 导出参数 ----------
# 字体设为 Arial
mpl.rcParams['font.family'] = 'Arial'
# 保证 PDF 文字在 Illustrator 可编辑
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# 保证 SVG 文字在 Illustrator 可编辑
mpl.rcParams['svg.fonttype'] = 'none'

# ---------- 数据读取 ----------
file_path = r"E:\100-科研\667_总数据版本\007层次聚类\插补后数据_cluster0805.csv"
df = pd.read_csv(file_path)

continuous_vars = [
    'Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
    'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
    'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
    'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
    'CK-MB', 'Creatinine', 'BUN', 'Alanine_aminotransferase',
    'Bilirubin', 'Albumin', 'Aspartate_aminotransferase'
]

# 按 PSS 四分位分组
df['PSS_group'] = pd.qcut(df['PSS'], 4, labels=['Quantile1', 'Quantile2', 'Quantile3', 'Quantile4'])

# Z-score 标准化
df_zscore = df[continuous_vars].apply(lambda x: (x - x.mean()) / x.std())
df_zscore['Group'] = df['PSS_group']

# 计算均值
df_group_mean = df_zscore.groupby('Group').mean().reset_index()

# 转长表
df_long = df_group_mean.melt(id_vars='Group', var_name='Variable', value_name='Standardized mean')

# ---------- 作图 ----------
plt.figure(figsize=(12, 6))
sns.barplot(data=df_long, x='Group', y='Standardized mean', hue='Variable', palette='tab20')

plt.axhline(0, color='black', linewidth=0.8)
plt.title('Variables in quantile PSS score', fontsize=14, fontweight='bold')
plt.ylabel('Standardized mean', fontsize=12)
plt.xlabel('PSS Quantile Subset clusters', fontsize=12)

# 图例改为两列
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Variables', ncol=2)

plt.tight_layout()

# ---------- 导出 ----------
# PDF：Illustrator 可编辑
plt.savefig(r"E:\100-科研\667_总数据版本\014PSS四分类\barplot_pss.pdf", format='pdf')
# 或 SVG：Illustrator 可编辑
plt.savefig(r"E:\100-科研\667_总数据版本\014PSS四分类\barplot_pss.svg", format='svg')

plt.show()



# In[3]:


import pandas as pd
import plotly.graph_objects as go

# 1. 读取数据
file_path = r"E:\100-科研\667_总数据版本\007层次聚类\插补后数据_cluster0805.csv"
df = pd.read_csv(file_path)

# 2. 定义左侧表型（cluster）和右侧PSS分位
df['PSS_group'] = pd.qcut(df['PSS'], 4, labels=['< 25%', '25%-50%', '50%-75%', '> 75%'])

# 3. 准备节点标签
cluster_labels = [f"Cluster {i}" for i in sorted(df['cluster'].unique())]
pss_labels = ['< 25%', '25%-50%', '50%-75%', '> 75%']
all_labels = cluster_labels + pss_labels

# 4. 节点编号
label_to_id = {label: idx for idx, label in enumerate(all_labels)}

# 5. 统计流量
links = []
for c in sorted(df['cluster'].unique()):
    for p in pss_labels:
        count = ((df['cluster'] == c) & (df['PSS_group'] == p)).sum()
        if count > 0:
            links.append({
                'source': label_to_id[f"Cluster {c}"],
                'target': label_to_id[p],
                'value': count
            })

# 6. 画 Sankey 图
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_labels,
        color=["#4daf4a", "#377eb8", "#ff7f00", "#f781bf"] +  # Cluster颜色
              ["#cccccc", "#999999", "#666666", "#333333"]   # PSS颜色
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        color="rgba(150,150,150,0.4)"  # 流线颜色
    )
)])

fig.update_layout(title_text="Cluster vs PSS Quantile", font_size=12)
fig.show()


# In[ ]:




