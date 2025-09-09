#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# 读取CSV数据
df = pd.read_csv(r"E:\100-科研\667_总数据版本\001整理好的版本_删除表头0804.csv", encoding='gbk')


# In[2]:


print(df.columns)


# In[3]:


# 备份原始数据
df_original = df.copy()

def outliers_proc(data, col_name, scale=1.5):
    """
    用 IQR 处理异常值
    data: pandas DataFrame
    col_name: 需要处理的列名
    scale: IQR 乘数
    """
    data_col = data[col_name]
    
    # 计算四分位数和 IQR
    Q1 = data_col.quantile(0.25)
    Q3 = data_col.quantile(0.75)
    IQR = Q3 - Q1
    
    # 计算异常值边界
    lower_bound = Q1 - (scale * IQR)
    upper_bound = Q3 + (scale * IQR)
    
    # 创建副本存储删除的数据
    deleted_data = data_col[(data_col < lower_bound) | (data_col > upper_bound)].copy()
    
    # 处理异常值
    data_col = data_col.apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
    
    return data_col, deleted_data

def remove_extreme_vals_z_score(data, col_name, sd_num=5):
    """
    用 Z-score 处理异常值
    data: pandas DataFrame
    col_name: 需要处理的列名
    sd_num: Z-score 阈值
    """
    data_col = data[col_name]
    
    # 计算均值和标准差
    mean = data_col.mean()
    sd = data_col.std()

    # 创建副本存储删除的数据
    deleted_data = data_col[abs((data_col - mean) / sd) > sd_num].copy()
    
    # 计算异常值边界
    data_col = data_col.apply(lambda x: x if abs((x - mean) / sd) <= sd_num else np.nan)
    
    return data_col, deleted_data

# 选择需要处理的列
columns_to_clean = ['Temperature', 'Heart_Rate',
       'Respiratory_Rate', 'MAP', 'P_F', 'PH', 'Base_Excess',
       'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
       'IL_6', 'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR',
       'D_dimer', 'Lactate', 'CK-MB', 'Creatinine', 'BUN', 'Ammonia',
       'Alanine_aminotransferase', 'Bilirubin', 'Albumin',
       'Aspartate_aminotransferase']

# 计算原始均值和中位数
original_mean = df_original[columns_to_clean].mean()
original_median = df_original[columns_to_clean].median()

# 复制数据，分别用于 IQR 和 Z-score 处理
df_iqr = df.copy()
df_z = df.copy()

# 创建字典保存删除的数据和数量
iqr_deleted_data = {}
z_score_deleted_data = {}
iqr_deleted_count = {}
z_score_deleted_count = {}

# IQR 方法
for col in columns_to_clean:
    if col in df_iqr.columns:
        df_iqr[col], deleted_data_iqr = outliers_proc(df_iqr, col)
        iqr_deleted_data[col] = deleted_data_iqr
        iqr_deleted_count[col] = len(deleted_data_iqr)

# Z-score 方法
for col in columns_to_clean:
    if col in df_z.columns:
        df_z[col], deleted_data_z = remove_extreme_vals_z_score(df_z, col)
        z_score_deleted_data[col] = deleted_data_z
        z_score_deleted_count[col] = len(deleted_data_z)

# 计算处理后的均值和中位数
iqr_mean = df_iqr[columns_to_clean].mean()
iqr_median = df_iqr[columns_to_clean].median()
z_mean = df_z[columns_to_clean].mean()
z_median = df_z[columns_to_clean].median()

# 计算均值和中位数的变化百分比
iqr_mean_change = (original_mean - iqr_mean) / original_mean * 100
iqr_median_change = (original_median - iqr_median) / original_median * 100
z_mean_change = (original_mean - z_mean) / original_mean * 100
z_median_change = (original_median - z_median) / original_median * 100

# 计算每个变量变成缺失值的比例
iqr_missing_percentage = df_iqr[columns_to_clean].isna().mean() * 100
z_missing_percentage = df_z[columns_to_clean].isna().mean() * 100

# 输出结果
print(f"IQR 处理后均值减少: {iqr_mean_change.mean():.2f}%")
print(f"IQR 处理后中位数减少: {iqr_median_change.mean():.2f}%")
print(f"Z-score 处理后均值减少: {z_mean_change.mean():.2f}%")
print(f"Z-score 处理后中位数减少: {z_median_change.mean():.2f}%")

# 输出每个变量变成缺失值的比例
print("\n每个变量变成缺失值的比例 (IQR方法):")
print(iqr_missing_percentage)

print("\n每个变量变成缺失值的比例 (Z-score方法):")
print(z_missing_percentage)

# 输出删除的数据数量
print("\n每个变量被删除的数据数量 (IQR方法):")
print(iqr_deleted_count)

print("\n每个变量被删除的数据数量 (Z-score方法):")
print(z_score_deleted_count)

# 保存删除的数据到CSV文件
for col, data in iqr_deleted_data.items():
    data.to_csv(f"IQR_removed_{col}.csv", index=False)
    
for col, data in z_score_deleted_data.items():
    data.to_csv(f"Z_score_removed_{col}.csv", index=False)



# In[4]:


non_numeric_cols = df_original[columns_to_clean].select_dtypes(exclude='number').columns
print("非数值列：", non_numeric_cols)


# In[4]:


df_z.to_csv("E:/100-科研/667_总数据版本/001清洗数据/Z_score_processed_data.csv", index=False)


# In[4]:


import pandas as pd
import numpy as np

# —— 备份原始数据
df_original = df.copy()

# —— 异常值处理函数
def outliers_proc(data, col_name, scale=1.5):
    data_col = data[col_name]
    Q1 = data_col.quantile(0.25)
    Q3 = data_col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (scale * IQR)
    upper_bound = Q3 + (scale * IQR)
    deleted_data = data_col[(data_col < lower_bound) | (data_col > upper_bound)].copy()
    data_col = data_col.apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
    return data_col, deleted_data

def remove_extreme_vals_z_score(data, col_name, sd_num=5):
    data_col = data[col_name]
    mean = data_col.mean()
    sd = data_col.std()
    deleted_data = data_col[abs((data_col - mean) / sd) > sd_num].copy()
    data_col = data_col.apply(lambda x: x if abs((x - mean) / sd) <= sd_num else np.nan)
    return data_col, deleted_data

# —— 选择需要处理的列
columns_to_clean = ['Temperature', 'Heart_Rate', 'Respiratory_Rate', 'MAP', 'P_F', 'PH', 'Base_Excess',
       'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin', 'IL_6',
       'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR',
       'D_dimer', 'Lactate', 'CK-MB', 'Creatinine', 'BUN', 'Ammonia',
       'Alanine_aminotransferase', 'Bilirubin', 'Albumin', 'Aspartate_aminotransferase']

# —— 复制数据
df_iqr = df.copy()
df_z = df.copy()

# —— 处理并记录删除数据数量
iqr_deleted_count = {}
z_score_deleted_count = {}

for col in columns_to_clean:
    if col in df_iqr.columns:
        df_iqr[col], deleted_iqr = outliers_proc(df_iqr, col)
        iqr_deleted_count[col] = len(deleted_iqr)
    if col in df_z.columns:
        df_z[col], deleted_z = remove_extreme_vals_z_score(df_z, col)
        z_score_deleted_count[col] = len(deleted_z)

# —— 计算删除比例
total_rows = df.shape[0]
iqr_deleted_percentage = {col: cnt/total_rows*100 for col, cnt in iqr_deleted_count.items()}
z_score_deleted_percentage = {col: cnt/total_rows*100 for col, cnt in z_score_deleted_count.items()}

# —— 合并成 DataFrame
iqr_summary = pd.DataFrame({
    "Variable": list(iqr_deleted_count.keys()),
    "Deleted_Count": list(iqr_deleted_count.values()),
    "Deleted_Percentage": list(iqr_deleted_percentage.values())
})

z_score_summary = pd.DataFrame({
    "Variable": list(z_score_deleted_count.keys()),
    "Deleted_Count": list(z_score_deleted_count.values()),
    "Deleted_Percentage": list(z_score_deleted_percentage.values())
})

# —— 导出 CSV
iqr_summary.to_csv(r"E:\100-科研\667_总数据版本\001清洗数据\IQR_deleted_summary.csv", index=False, encoding="utf-8-sig")
z_score_summary.to_csv(r"E:\100-科研\667_总数据版本\001清洗数据\Z_score_deleted_summary.csv", index=False, encoding="utf-8-sig")

# —— 输出结果查看
print("IQR 删除统计:")
print(iqr_summary)
print("\nZ-score 删除统计:")
print(z_score_summary)


# In[ ]:





# In[ ]:




