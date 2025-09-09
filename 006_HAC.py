#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import umap

# 读取数据
df = pd.read_csv(r"E:\100-科研\667_总数据版本\003对数转换\scaled_for_clustering_with_ID.csv")

# 设置 ID 列
id_col = 'id'

# 聚类变量


CLUSTERING_LABs = [
    'Age', 'Heart_Rate', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
    'Hemoglobin', 'Platelet', 'BUN','INR', 'D_dimer', 'PH',
    'CK-MB', 'Alanine_aminotransferase', 'White_blood_cell','PSS'
]

# 取聚类数据
X = df[CLUSTERING_LABs].values

# -----------------------------



# # 进行层次聚类

# In[4]:


# 1. Agglomerative Clustering
# -----------------------------
model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')

cluster_labels = model.fit_predict(X)
df['cluster'] = cluster_labels + 1  # 使 cluster 从 1 开始

# 输出每个聚类的样本数量
print("每个聚类的样本数量：")
print(df['cluster'].value_counts().sort_index())


C = 4
lable_color = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}
               
lable_annotation = {1:'Subphenotype I',
                    2:'Subphenotype II',
                    3:'Subphenotype III',
                    4:'Subphenotype IV',
                    }             
                              
# rename labels
clust_label_map = {
        1:4,
        2:2,
        3:3,
        4:1,
        }
new_labels = []
for i in labels:
    new_labels.append(clust_label_map[i])
labels = np.array(new_labels)



# # 改标签

# In[5]:


# 假设已经聚类完成，并将 cluster 从 1 开始赋值
df['cluster'] = cluster_labels + 1

# 自定义标签映射（你想重命名）
clust_label_map = {
    1: 4,
    2: 2,
    3: 3,
    4: 1,
}

# 应用映射到 DataFrame 的 'cluster' 列
df['cluster'] = df['cluster'].map(clust_label_map)

# 输出结果检查
print("每个聚类的样本数量（重命名后）：")
print(df['cluster'].value_counts().sort_index())


# In[6]:


# 保存含聚类标签的新数据到 CSV
#df.to_csv(r"E:\100-科研\667_总数据版本\007层次聚类\clustered_with_labels.csv", index=False)
#df.to_csv(r"E:\100-科研\667_总数据版本\007层次聚类\changed_labels.csv", index=False)


# # 可视化

# In[7]:


# 自定义颜色映射
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}
row_colors = df['cluster'].map(color_map)

# -----------------------------
# 2. clustermap 热图
# -----------------------------
sns.clustermap(
    df[CLUSTERING_LABs],
    row_cluster=True,
    col_cluster=False,
    row_colors=row_colors,
    figsize=(12, 10),
    cmap='coolwarm',
    standard_scale=1
)
plt.title("Cluster Map with Row Dendrogram", pad=80)
plt.show()


# In[7]:


# 3. Dendrogram
# -----------------------------
linked = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# # 改变字体

# In[13]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用 Arial 字体
plt.rcParams['font.size'] = 12 # 增大基础字体大小
plt.rcParams['pdf.fonttype'] = 42  # 确保在 AI 中可编辑
plt.rcParams['ps.fonttype'] = 42   # 确保在 AI 中可编辑

# 自定义颜色映射
color_map = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}
row_colors = df['cluster'].map(color_map)

# -----------------------------
# 2. clustermap 热图
# -----------------------------
# 创建 ClusterGrid 对象
g = sns.clustermap(
    df[CLUSTERING_LABs],
    row_cluster=True,
    col_cluster=False,
    row_colors=row_colors,
    figsize=(12, 10),
    cmap='coolwarm',
    standard_scale=1
)

# 设置标题（使用更大的字体）
#g.fig.suptitle("Cluster Map with Row Dendrogram", y=0.92, fontsize=14, fontname='Arial')

# 增大轴标签字体
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(), 
    fontsize=13,  # 增大列标签字体
    fontname='Arial'
)

# 增大树状图标签字体（如果显示）
if g.ax_row_dendrogram:
    for text in g.ax_row_dendrogram.get_yticklabels():
        text.set_fontsize(10)
        text.set_fontname('Arial')

# 增大颜色条标签字体
if g.cax:
    g.cax.yaxis.set_tick_params(labelsize=11)
    for text in g.cax.get_yticklabels():
        text.set_fontname('Arial')

# 导出为 PDF 文件（确保在 AI 中可编辑）
g.savefig(
    r"E:\100-科研\667_总数据版本\014写作和结果汇总\02图表0811单个\clustermap_visualization.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300
)

# 显示图形（可选）
plt.show()


# In[14]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
plt.rcParams['font.size'] = 12  # 增大基础字体大小
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # X轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 10  # Y轴刻度标签大小
plt.rcParams['pdf.fonttype'] = 42  # 确保在AI中可编辑
plt.rcParams['ps.fonttype'] = 42   # 确保在AI中可编辑

# -----------------------------
# 3. Dendrogram
# -----------------------------
linked = linkage(X, method='ward')

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制树状图
dn = dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False,
               no_labels=True)

# 设置标题和标签（使用Arial字体）
plt.title('Hierarchical Clustering Dendrogram', fontname='Arial', fontsize=14)
plt.xlabel('Samples', fontname='Arial', fontsize=13)
plt.ylabel('Distance', fontname='Arial', fontsize=13)

# 调整刻度标签字体
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
    label.set_fontsize(10)
    
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_fontsize(10)

# 保存为PDF（确保在AI中可编辑）
plt.savefig(
    r"E:\100-科研\667_总数据版本\014写作和结果汇总\02图表0811单个\hierarchical_dendrogram.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300
)

# 显示图形
plt.show()


# In[8]:


# -----------------------------
# 4. UMAP 降维 + 聚类颜色
# -----------------------------
reducer = umap.UMAP(random_state=123)
embedding = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
for c in sorted(df['cluster'].unique()):
    plt.scatter(
        embedding[df['cluster'] == c, 0],
        embedding[df['cluster'] == c, 1],
        label=f'Cluster {c}',
        color=color_map[c],
        alpha=0.8
    )

plt.title("UMAP Projection with Cluster Labels")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.show()


# In[ ]:




