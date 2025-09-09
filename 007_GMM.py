#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取预处理数据（确保已 log 和标准化）
df_scaled_df = pd.read_csv(r"E:\100-科研\667_总数据版本\003对数转换\scaled_for_clustering_with_ID.csv")
X = df_scaled_df.drop(columns=['id']).values

range_n_clusters = range(2, 9)  # 尝试聚类个数：2~8
bic_scores, aic_scores = [], []
models = []
median_probs, iqr_probs = [], []

for n_clusters in range_n_clusters:
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',
        random_state=3,
        init_params='kmeans',
        n_init=10  # 多次初始化以保证稳定性
    )
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))
    models.append(gmm)

    probs = gmm.predict_proba(X).max(axis=1)
    median_prob = np.median(probs)
    iqr_prob = np.percentile(probs, 75) - np.percentile(probs, 25)
    median_probs.append(median_prob)
    iqr_probs.append(iqr_prob)

    print(f"Clusters: {n_clusters} | BIC: {bic_scores[-1]:.2f}, AIC: {aic_scores[-1]:.2f}, "
          f"Median Prob: {median_prob:.3f}, IQR: {iqr_prob:.3f}")

# 绘图：BIC 和 AIC 曲线
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, bic_scores, label='BIC', marker='o')
plt.plot(range_n_clusters, aic_scores, label='AIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('GMM Model Selection using BIC and AIC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 选出BIC最小的模型
best_n = range_n_clusters[np.argmin(bic_scores)]
best_gmm = models[np.argmin(bic_scores)]
print(f"\n✅ Best cluster number by BIC: {best_n}")

# 聚类标签
labels = best_gmm.predict(X)
df_scaled_df['GMM_cluster'] = labels




# In[5]:





# In[4]:


# 输出结果检查
#print("每个聚类的样本数量（重命名后）：")
#print(df_scaled_df['GMM_cluster'].value_counts().sort_index())


# In[6]:


C = 4
lable_color = {1:'#79A3D9', 2:'#7B967A', 3:'#F9C77E', 4:'#CE4257'}
               
lable_annotation = {1:'Subphenotype I',
                    2:'Subphenotype II',
                    3:'Subphenotype III',
                    4:'Subphenotype IV',
                    }             
                              
# rename labels
clust_label_map = {
    0: 1,
    1: 4,
    2: 2,
    3: 3
}

new_labels = [clust_label_map[i] for i in labels]
labels = np.array(new_labels)


# In[7]:


# 添加标签到原始数据
df_scaled_df['GMM_cluster'] = labels


# In[9]:


# 保存带聚类标签的数据
#df_scaled_df.to_csv(r"E:\100-科研\667_总数据版本\005GMM\002_GMM_cluster_with_labels.csv", index=False)


# In[ ]:




