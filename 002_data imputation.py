#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# 读取CSV数据
data = pd.read_csv(r"E:\100-科研\667_总数据版本\001清洗数据\Z_score_processed_data.csv", encoding='gbk')


# In[2]:


print(data.columns)


# # 查看缺失率

# In[3]:


# 显示所有行（默认显示前后几行，中间省略）
pd.set_option('display.max_rows', None)

# 显示所有列（如果你想查看所有列也可以设这个）
pd.set_option('display.max_columns', None)

# 显示完整内容而不省略
pd.set_option('display.max_colwidth', None)


# In[4]:


# 每行的缺失率
row_missing_rate = data.isnull().sum(axis=1) / data.shape[1]

# 假设第一列是病人 ID
row_missing_df = pd.DataFrame({
    'ID': data.iloc[:, 0],
    'Missing_Rate': row_missing_rate
})

# 显示所有病人缺失率
print(row_missing_df)


# In[5]:


# 按照缺失率从高到低排序
row_missing_df_sorted = row_missing_df.sort_values('Missing_Rate', ascending=False)

# 显示结果
print(row_missing_df_sorted)


# In[6]:


# 每列的缺失率（缺失值数 / 总行数）
col_missing_rate = data.isnull().sum() / data.shape[0]

# 如果你想以 DataFrame 的形式看
col_missing_df = col_missing_rate.reset_index()
col_missing_df.columns = ['Variable', 'Missing_Rate']
# 显示所有病人缺失率
print(col_missing_df)


# In[7]:


# 导出到 CSV 文件
col_missing_df.to_csv(r"E:\100-科研\667_总数据版本\001清洗数据\col_missing_rate.csv",
                      index=False, encoding="utf-8-sig")


# In[7]:


# 找出缺失率大于 30% 的变量
high_missing_vars = col_missing_df[col_missing_df['Missing_Rate'] > 0.3]

# 显示结果
print(high_missing_vars)


# # 进行插补

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


# In[9]:


print(data.columns)


# In[10]:


continuous_cols = [
    'Age', 'BMI',  'Temperature', 'Heart_Rate',
       'Respiratory_Rate', 'MAP', 'P_F', 'PH', 'Base_Excess',
       'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
        'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR',
       'D_dimer', 'Lactate', 'CK-MB', 'Creatinine', 'BUN', 
       'Alanine_aminotransferase', 'Bilirubin', 'Albumin',
       'Aspartate_aminotransferase', 'Blood_Purification_Time', 'Mv_Time',
       'PSS', 'ICU_LOS', 'LOS', 'Survival_Time',
]

high_missing_cols = [
    'IL_6', 'Ammonia'
]

target_cols=[
    'Age', 'BMI',  'Temperature', 'Heart_Rate',
       'Respiratory_Rate', 'MAP', 'P_F', 'PH', 'Base_Excess',
       'White_blood_cell', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
        'Red_blood_cell', 'Hemoglobin', 'Platelet', 'PT', 'APTT', 'INR',
       'D_dimer', 'Lactate', 'CK-MB', 'Creatinine', 'BUN', 
       'Alanine_aminotransferase', 'Bilirubin', 'Albumin',
       'Aspartate_aminotransferase'
]
all_variable = target_cols + high_missing_cols

categorical_cols = [
    '28_day_mortality', 'Gender','preparation_time', 'Blood_Purification_Modality', 'HVHF',
       'Anticoagulation_Strategy', 'Vascular_Access_Site',
       'Thrombosis_CatheterSite', 'Sedation', 'Low_Blood_Pressure',
       'Nutrition_Method', 'Consciousness', 'ECMO', 'Mechanical_ventilation',
       'Epinephrine', 'Norepinephrine', 'Dopamine'
] 


# In[11]:


# 2. 选择连续变量子集


# 3. 提取连续变量数据
df_cont = data[all_variable]

# 4. 计算皮尔逊相关系数矩阵
corr_matrix = df_cont.corr(method='pearson')  # 或 'spearman'，根据分布情况选择

# 5. 可视化：热图
plt.figure(figsize=(18, 16))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Pearson Correlation of Continuous Variables", fontsize=18)
plt.tight_layout()
plt.show()


# In[12]:


# 合并所有变量
all_cols = continuous_cols + categorical_cols

# 计算缺失比例（百分比形式）
missing_percent = data[all_cols].isnull().mean() * 100

# 整理成DataFrame格式
missing_df = pd.DataFrame({
    'Variable': missing_percent.index,
    'Missing_Percent': missing_percent.values
})

# 保留两位小数
missing_df['Missing_Percent'] = missing_df['Missing_Percent'].round(2)

# 导出为CSV
output_path = r"E:\100-科研\667_总数据版本\001清洗数据\变量缺失比例2.csv"
#missing_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("缺失比例计算完成，文件已保存至：", output_path)


# In[13]:


# 定义快速插补方法
methods = {
    'pmm': KNNImputer(n_neighbors=5),
    'cart': IterativeImputer(estimator=DecisionTreeRegressor(max_depth=5), 
                            max_iter=10, random_state=42),
    'midastouch': KNNImputer(n_neighbors=10, weights='distance'),
    'rf': IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, 
                                                         n_jobs=-1, 
                                                         max_depth=5),
                         max_iter=10, random_state=42)
}

# 生成20个数据集（每个方法5个）
imputed_datasets = {}
for method_name, imputer in methods.items():
    print(f"正在处理 {method_name}...")
    method_dfs = []
    for i in range(5):
        # 设置不同随机种子保证多样性
        if hasattr(imputer, 'random_state'):
            imputer.random_state = i
            
        imputed = imputer.fit_transform(data[continuous_cols])
        imputed_df = pd.DataFrame(imputed, columns=continuous_cols)
        method_dfs.append(imputed_df)
    imputed_datasets[method_name] = method_dfs
    


# In[15]:


# 合并数据集（每个方法生成均值/中位数合并）
merged_results = {}
for method, dfs in imputed_datasets.items():
    # 使用三维数组加速计算
    cube = np.stack([df.values for df in dfs], axis=2)
    
    # 均值合并
    mean_merged = pd.DataFrame(np.mean(cube, axis=2), 
                              columns=continuous_cols)
    
    # 中位数合并
    median_merged = pd.DataFrame(np.median(cube, axis=2), 
                                columns=continuous_cols)
    
    merged_results[f"{method}_mean"] = mean_merged
    merged_results[f"{method}_median"] = median_merged


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import joypy
import os

# 设置保存路径（可自定义）
save_dir = r"E:\100-科研\667_总数据版本\002数据插补\JoyPlots_all_vars"
os.makedirs(save_dir, exist_ok=True)

# 获取所有插补后的变量名（从任意一个插补结果中获取）
all_vars = merged_results[list(merged_results.keys())[0]].columns.tolist()

# 设置颜色
colors = [plt.cm.tab20(i % 20) for i in range(len(merged_results))] + ['#000000']

# 循环绘图
for var in all_vars:
    try:
        if var not in data.columns:
            print(f"⚠️ 变量 {var} 在原始数据中不存在，跳过。")
            continue

        original = data[var].dropna()
        joy_data = []

        for name, df in merged_results.items():
            if var in df.columns:
                temp_df = df[[var]].copy()
                temp_df['Method'] = name.replace('_', ' ')
                joy_data.append(temp_df)

        # 添加原始数据
        original_df = pd.DataFrame({var: original})
        original_df['Method'] = 'z_original data'
        joy_data.append(original_df)

        # 合并数据
        plot_df = pd.concat(joy_data).reset_index(drop=True)

        # Joyplot 要求变量是数字或可枚举的类别（尝试强制转换为str用于分类变量）
        if plot_df[var].dtype == 'object' or str(plot_df[var].dtype).startswith('category'):
            plot_df[var] = plot_df[var].astype(str)

        # 绘图
        plt.figure(figsize=(14, 10), dpi=300)
        fig, axes = joypy.joyplot(
            plot_df,
            by='Method',
            column=var,
            color=colors,
            fade=True,
            overlap=2.5,
            linewidth=1.5,
            alpha=0.7,
            background='white',
            grid=True,
            title=f'{var}',
            xlabelsize=12,
            range_style='all',
            ylabelsize=10,
            hist=False
        )

        # 高亮原始数据线
        axes[-1].lines[0].set_color('black')
        axes[-1].collections[0].set_color('black')
        axes[-1].collections[0].set_alpha(0.3)

        plt.subplots_adjust(top=0.92, hspace=-0.3)
        plt.xlabel(var, fontsize=12, labelpad=10)
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f"{var}_joyplot.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ {var} 图已保存: {save_path}")

    except Exception as e:
        print(f"❌ 绘图失败：{var}，错误：{e}")


# In[15]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# 分类变量插补（使用一种方法即可）
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputed = cat_imputer.fit_transform(data[categorical_cols])
cat_imputed_df = pd.DataFrame(cat_imputed, columns=categorical_cols)


# In[16]:


# 合并分类变量
pmm_mean = merged_results['pmm_mean']
pmm_mean_with_categorical = pd.concat([pmm_mean, cat_imputed_df], axis=1)

# 导出为 CSV 文件
#pmm_mean_with_categorical.to_csv(r"E:\100-科研\667_总数据版本\002数据插补\pmm_mean_with_categorical_0804.csv", index=False)

print("pmm_mean_0804 已成功导出！")


# In[24]:


# 合并分类变量
pmm_median = merged_results['pmm_median']

# 导出为 CSV 文件
pmm_median.to_csv(r"E:\100-科研\667_总数据版本\002数据插补\pmm_median0814.csv", index=False)


# In[ ]:




