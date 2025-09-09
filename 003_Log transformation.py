#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[22]:


# load data

data = pd.read_csv(r"E:\100-科研\667_总数据版本\002数据插补\pmm_mean_with_categorical_0804.csv")
data.head()
print(data.columns)

TARGET_LABs = ['Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
       'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
       'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
       'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
       'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
       'Bilirubin', 'Albumin', 'Aspartate_aminotransferase','PSS'
    
        ]

CLUSTERING_COLS = TARGET_LABs


# In[23]:


TARGET_LABs = ['Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
       'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
       'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
       'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
       'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
       'Bilirubin', 'Albumin', 'Aspartate_aminotransferase','PSS'
    
        ]

CLUSTERING_COLS = TARGET_LABs


# In[24]:


from scipy.stats import skew, kurtosis

# 筛选出要分析的变量列
df_selected = data[CLUSTERING_COLS]

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


# In[25]:


from scipy.stats import shapiro

shapiro_results = []

for col in CLUSTERING_COLS:
    # Shapiro-Wilk 检验
    stat, p_value = shapiro(data[col].dropna())
    shapiro_results.append({'Variable': col, 'Shapiro_p': p_value})

# 汇总成 DataFrame
shapiro_df = pd.DataFrame(shapiro_results)

# 标记是否拒绝正态性假设（p < 0.05）
shapiro_df['Normal_Distribution'] = shapiro_df['Shapiro_p'] >= 0.05

# 显示结果
print(shapiro_df.sort_values('Shapiro_p'))


# ![image.png](attachment:image.png)

# In[28]:


log_transform_vars = [
    'Bilirubin',
    'D_dimer',
    'Alanine_aminotransferase',
    'Aspartate_aminotransferase',
    ''
    'CK-MB',
    'Creatinine',
    'BUN',
    'Lactate',
    'Procalcitonin',
    'APTT',
    'PT',
    'BMI',
    'C_reactive_protein',
    'P_F',
    'White_blood_cell',
    'Platelet'
]


# In[29]:


# 对偏态变量进行 log1p 转换（log(1 + x)）
data_log_transformed = data.copy()
for col in log_transform_vars:
    data_log_transformed[col] = np.log1p(data_log_transformed[col])


# In[30]:


# 查看转换后是否改善偏态
transformed_skewness = data_log_transformed[log_transform_vars].skew()
print(transformed_skewness.sort_values(ascending=False))


# In[17]:


from sklearn.preprocessing import PowerTransformer

# 使用 Yeo-Johnson 变换（可用于含 0 或负值的分布）
pt = PowerTransformer(method='yeo-johnson')

# 对 INR 和 PT 做变换
data_log_transformed[['INR','PT']] = pt.fit_transform(data_log_transformed[['INR', 'PT']])


# In[18]:


print("转换后偏度（Skewness）:")
print(data_log_transformed[['INR', 'PT']].skew())


# # 以下是把转换后的变量和没有整合的导出为同一个表格

# In[31]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# 读取数据
data = pd.read_csv(r"E:\100-科研\667_总数据版本\002数据插补\pmm_mean_with_categorical_0804.csv")

# 所有聚类变量
TARGET_LABs = ['Age', 'BMI', 'Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP',
       'P_F', 'PH', 'Base_Excess', 'White_blood_cell', 'Neutrophil',
       'C_reactive_protein', 'Procalcitonin', 'Red_blood_cell',
       'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR', 'D_dimer', 'Lactate',
       'CK-MB', 'Creatinine', 'BUN',  'Alanine_aminotransferase',
       'Bilirubin', 'Albumin', 'Aspartate_aminotransferase', 'PSS']

# 定义需要进行 log1p 转换的变量
log_transform_vars = [
    'Bilirubin', 'D_dimer', 'Alanine_aminotransferase', 'Aspartate_aminotransferase',
    'INR', 'CK-MB', 'Creatinine', 'BUN', 'Lactate', 'Procalcitonin',
    'APTT', 'PT', 'BMI', 'C_reactive_protein', 'P_F',
     'White_blood_cell', 'Platelet'
]

# 拷贝原始数据
data_transformed = data.copy()

# ---------- log1p 转换 ----------
# 对除 INR 和 PT 外的变量做 log1p（这些变量用 Yeo-Johnson）
log1p_vars = [col for col in log_transform_vars if col not in ['INR', 'PT']]
for col in log1p_vars:
    data_transformed[col] = np.log1p(data_transformed[col])

# ---------- Yeo-Johnson 转换 ----------
pt = PowerTransformer(method='yeo-johnson')
data_transformed[['INR', 'PT']] = pt.fit_transform(data[['INR', 'PT']])

# ---------- 最终聚类用变量（保持原顺序） ----------
final_clustering_df = data_transformed[TARGET_LABs]

# ---------- 导出 ----------
output_path = r"E:\100-科研\667_总数据版本\003对数转换\LOG_0804.csv"

final_clustering_df.to_csv(output_path, index=False)
print(f"✅ 处理后的数据已保存到：{output_path}")


# In[ ]:




