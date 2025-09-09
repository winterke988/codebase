#!/usr/bin/env python
# coding: utf-8

# In[1]:


import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# In[2]:


# 1. 提取用于UMAP的输入数据（已经标准化的）
df_scaled_df = pd.read_csv(r"E:\100-科研\667_总数据版本\009用于做UMAP的数据\scaled_with_all_labels.csv")
X = df_scaled_df.drop(columns=['id']).values

print(df_scaled_df.columns)


# In[6]:


import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 提取 UMAP 输入数据（去掉 id 和聚类标签）
umap_features = df_scaled_df.iloc[:, :15].values  # 前12列为数值变量

# 2. UMAP 降维
reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42)
embedding = reducer.fit_transform(umap_features)

# 3. 加入降维结果
df_scaled_df['UMAP1'] = embedding[:, 0]
df_scaled_df['UMAP2'] = embedding[:, 1]

# 4. 绘制三图并列对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# (1) 共识聚类
sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='cluster',
    palette='Set2',
    ax=axes[0]
)
axes[0].set_title('Agglomerative Clustering')

# (2) GMM
sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='GMM_cluster',
    palette='Set2',
    ax=axes[1]
)
axes[1].set_title('GMM Clustering')

# (3) LPA

sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='Class',
    palette='Set2',
    ax=axes[2]
)
axes[2].set_title('LPA Clustering')

# 统一图例位置
for ax in axes:
    ax.legend(title='Cluster', loc='best')

plt.tight_layout()
plt.show()


# In[3]:


import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 提取 UMAP 输入数据（去掉 id 和聚类标签）
umap_features = df_scaled_df.iloc[:, :15].values  # 替换为你的特征列

# 2. UMAP 降维（只需做一次）
reducer = umap.UMAP(n_neighbors=10, min_dist=0.35, random_state=42)
embedding = reducer.fit_transform(umap_features)
df_scaled_df['UMAP1'] = embedding[:, 0]
df_scaled_df['UMAP2'] = embedding[:, 1]


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}


plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='cluster',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('Agglomerative hierarchical clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Cluster', loc='best')

#plt.savefig("Agglomerative Clustering_UMAP.png", dpi=1200)  # 保存高分辨率图像
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}


plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='GMM_cluster',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('GMM Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Cluster', loc='best')

#plt.savefig("GMM_clustering_UMAP.png", dpi=1200)  # 保存高分辨率图像
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}


plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='Class',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('LPA Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Cluster', loc='best')

#plt.savefig("LPA Clustering_UMAP.png", dpi=1200)  # 保存高分辨率图像
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# === 设置全局字体为 Arial，并保持在 Illustrator 可编辑 ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'  # SVG 不转曲线，保留文字
plt.rcParams['pdf.fonttype'] = 42      # PDF 保留文字可编辑

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}

plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='cluster',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('Agglomerative Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

# 替换 legend 标签为罗马数字
new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Subphenotype', loc='best')

# 保存矢量格式（可编辑字体）
plt.savefig(r"E:\100-科研\667_总数据版本\009用于做UMAP的数据\Agglomerative_Clustering_UMAP2.pdf",format="pdf", bbox_inches="tight",
    dpi=300 )  # Illustrator 推荐用 SVG
# plt.savefig("Agglomerative_Clustering_UMAP.pdf", dpi=300, format="pdf")  # 备用 PDF

plt.show()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# === 设置全局字体为 Arial，并保持在 Illustrator 可编辑 ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'  # SVG 不转曲线，保留文字
plt.rcParams['pdf.fonttype'] = 42      # PDF 保留文字可编辑

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}

plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='GMM_cluster',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('GMM Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

# 替换 legend 标签为罗马数字
new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Subphenotype', loc='best')

# 保存矢量格式（可编辑字体）
plt.savefig(r"E:\100-科研\667_总数据版本\009用于做UMAP的数据\GMM_Clustering_UMAP2.pdf",format="pdf", bbox_inches="tight",
    dpi=300 )  # Illustrator 推荐用 SVG
# plt.savefig("Agglomerative_Clustering_UMAP.pdf", dpi=300, format="pdf")  # 备用 PDF

plt.show()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# === 设置全局字体为 Arial，并保持在 Illustrator 可编辑 ===
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'  # SVG 不转曲线，保留文字
plt.rcParams['pdf.fonttype'] = 42      # PDF 保留文字可编辑

# 自定义颜色映射（类别为整数）
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}

# 罗马数字标签映射
roman_labels = {1:'I', 2:'II', 3:'III', 4:'IV'}

plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=df_scaled_df,
    x='UMAP1', y='UMAP2',
    hue='Class',
    palette=color_map,
    hue_order=[1, 2, 3, 4],
    legend='full'
)

plt.title('LPA Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

handles, labels = scatter.get_legend_handles_labels()

# 替换 legend 标签为罗马数字
new_labels = []
for l in labels:
    try:
        key = int(l)
        new_labels.append(roman_labels.get(key, l))
    except:
        new_labels.append(l)

plt.legend(handles=handles, labels=new_labels, title='Subphenotype', loc='best')

# 保存矢量格式（可编辑字体）
plt.savefig(r"E:\100-科研\667_总数据版本\009用于做UMAP的数据\LPA_Clustering_UMAP2.pdf",format="pdf", bbox_inches="tight",
    dpi=300 )  # Illustrator 推荐用 SVG
# plt.savefig("Agglomerative_Clustering_UMAP.pdf", dpi=300, format="pdf")  # 备用 PDF

plt.show()


# In[ ]:




