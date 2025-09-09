#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

#from _clust_charaterization import two_clust_compare
import umap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import datetime
from joblib import dump, load
import pickle as pkl


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.ensemble import RandomForestClassifier
import shap


# In[3]:


# load data

df = pd.read_csv(r"E:\100-科研\667_总数据版本\003对数转换\LOG_0804.csv")
df.head()
print(df.columns)
TARGET_LABs = ['Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
       'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
       'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
       'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
       'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
       'Bilirubin', 'Albumin', 'Aspartate_aminotransferase','PSS'
    
        ]


# In[4]:


TARGET_LABs = ['Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
       'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
       'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
       'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
       'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
       'Bilirubin', 'Albumin', 'Aspartate_aminotransferase','PSS'
    
        ]



# In[5]:


from scipy.stats import skew, kurtosis

# 筛选出要分析的变量列
df_selected = df[TARGET_LABs]

# 计算偏度和峰度
skewness = df_selected.skew()
kurt = df_selected.kurtosis()

# 合并成一个DataFrame
distribution_df = pd.DataFrame({
    'Skewness': skewness,
    'Kurtosis': kurt
})

# 查看偏态信息
print(distribution_df.sort_values('Skewness', ascending=False))


# In[6]:


df = df[TARGET_LABs].dropna()


# In[7]:


# === Step 1. 标准化 & 因子分析 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[8]:


fa = FactorAnalyzer(rotation='varimax')
fa.fit(X_scaled)


# In[9]:


# 1.1 碎石图选择因子数
ev, _ = fa.get_eigenvalues()
plt.figure(figsize=(8,4))
plt.plot(range(1, len(ev)+1), ev, marker='o')
plt.axhline(1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()


# In[10]:


# 1.2 设定合适因子数，重新拟合
n_factors = 7 # 可根据碎石图选择
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(X_scaled)
# 1.3 因子载荷
loadings = pd.DataFrame(fa.loadings_, index=TARGET_LABs, columns=[f'Factor{i+1}' for i in range(n_factors)])
print("因子载荷矩阵：")
print(loadings)


# In[11]:


# 导出因子载荷矩阵到 CSV
loadings.to_csv(r"E:\100-科研\667_总数据版本\003正交因子\因子载荷矩阵.csv", encoding="utf-8-sig")  # utf-8-sig 避免中文乱码


# In[25]:


# 1.3 因子载荷
loadings = pd.DataFrame(fa.loadings_, index=TARGET_LABs, columns=[f'Factor{i+1}' for i in range(n_factors)])
print("因子载荷矩阵：")
print(loadings)


# In[26]:


# loadings 是你的因子载荷矩阵 DataFrame，行是变量，列是因子

threshold = 0.5 # 载荷阈值

# 用来存放每个因子对应的高载荷变量
factor_vars = {}

for factor in loadings.columns:
    # 选择该因子中载荷绝对值大于阈值的变量
    selected_vars = loadings.index[loadings[factor].abs() > threshold].tolist()
    factor_vars[factor] = selected_vars

# 打印结果
for factor, vars_list in factor_vars.items():
    print(f"{factor} 中载荷绝对值 > {threshold} 的变量有：")
    if vars_list:
        for var in vars_list:
            loading_value = loadings.loc[var, factor]
            print(f"  {var}: {loading_value:.3f}")
    else:
        print("  无符合条件的变量")
    print()


# In[44]:


# loadings 是你的因子载荷矩阵 DataFrame，行是变量，列是因子

threshold = 0.4 # 载荷阈值

# 用来存放每个因子对应的高载荷变量
factor_vars = {}

for factor in loadings.columns:
    # 选择该因子中载荷绝对值大于阈值的变量
    selected_vars = loadings.index[loadings[factor].abs() > threshold].tolist()
    factor_vars[factor] = selected_vars

# 打印结果
for factor, vars_list in factor_vars.items():
    print(f"{factor} 中载荷绝对值 > {threshold} 的变量有：")
    if vars_list:
        for var in vars_list:
            loading_value = loadings.loc[var, factor]
            print(f"  {var}: {loading_value:.3f}")
    else:
        print("  无符合条件的变量")
    print()


# In[13]:


# Step 1: 计算每个因子的特征值（Eigenvalue）
eigenvalues = (loadings ** 2).sum(axis=0)

# Step 2: 解释方差比例（%）
explained_variance_ratio = eigenvalues / eigenvalues.sum()

# Step 3: 累积解释方差（%）
cumulative_variance = explained_variance_ratio.cumsum()

# 打印结果
result = pd.DataFrame({
    'Eigenvalue': eigenvalues,
    'ExplainedVariance(%)': explained_variance_ratio * 100,
    'Cumulative(%)': cumulative_variance * 100
})

print("\n因子特征值及解释方差：")
print(result.round(2))


# In[14]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 取出分析变量
data_vars = df[TARGET_LABs].copy()

# 标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_vars)
data_scaled_df = pd.DataFrame(data_scaled, columns=TARGET_LABs)

# 计算相关系数矩阵（Pearson）
corr_matrix = data_scaled_df.corr()

# 打印相关系数矩阵
print("变量相关系数矩阵：")
print(corr_matrix)

# 可视化相关系数矩阵（热力图）
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            square=True, linewidths=0.5, annot_kws={"size":7})
plt.title("变量相关系数矩阵热力图")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# In[12]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 选取目标变量
data_selected = df[TARGET_LABs]

# 标准化（Z-score）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)
data_scaled = pd.DataFrame(data_scaled, columns=TARGET_LABs)

# 计算相关系数矩阵
corr_matrix = data_scaled.corr()

# 画热力图
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={'shrink':.75}, vmax=1, vmin=-1, linewidths=0.5)
plt.title('Variables Correlation Heatmap (Standardized Data)')
plt.tight_layout()
plt.show()

# 打印相关系数绝对值 > 0.6 的变量对（去除自相关）
print("相关系数绝对值大于 0.6 的变量对：")
threshold = 0.6
# 只看矩阵上三角（不含对角线）
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
)

high_corr = corr_pairs.stack().reset_index()
high_corr.columns = ['Variable1', 'Variable2', 'Correlation']

high_corr_filtered = high_corr[high_corr['Correlation'].abs() > threshold]

for _, row in high_corr_filtered.iterrows():
    print(f"{row['Variable1']} 与 {row['Variable2']} 相关系数 = {row['Correlation']:.3f}")


# In[13]:


plt.figure(figsize=(16, 14))  # 图像更大
sns.heatmap(
    corr_matrix,
    annot=True,          # 显示数值
    fmt=".2f",           # 保留两位小数
    cmap='coolwarm',
    square=True,
    cbar_kws={'shrink': .75},
    vmax=1, vmin=-1,
    linewidths=0.5,
    annot_kws={"size": 8}  # 调整字体大小
)

# 旋转标签，防止挤在一起
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.title('Variables Correlation Heatmap (Standardized Data)', fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:




