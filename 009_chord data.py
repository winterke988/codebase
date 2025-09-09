#!/usr/bin/env python
# coding: utf-8

# # GMM生成和弦图数据

# In[ ]:


# 定义系统映射（用你的 system_mapping）
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
    'CK-MB': 'Circulation'
}



REVERSE_DIRECTION_COLS = {
    'Albumin',
    'Hemoglobin',
    'Platelet',
    'Red_blood_cell',
    'MAP',
    'PH',
    'P_F',
    'Base_Excess',
}




import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv(r"E:\100-科研\667_总数据版本\005GMM\插补后数据_GMMcluster0805.csv")

cluster_col = 'GMM_cluster'
C = data[cluster_col].nunique()  # 聚类总数，例如3类或4类

# 构建 system to variable list（和最开始的 LAB_VARS 等价）
system_vars = {}
for var, system in system_mapping.items():
    system_vars.setdefault(system, []).append(var)


chord_data_pct = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]
    Nc = len(temp_data)

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue  # 某些变量可能缺失
            if var not in REVERSE_DIRECTION_COLS:
                n = len(temp_data[temp_data[var] > np.nanmedian(data[var])])
            else:
                n = len(temp_data[temp_data[var] < np.nanmedian(data[var])])
            value += 100 * n / Nc

        chord_data_pct.append([f'Cluster {c}', system, value])

df_chord_pct = pd.DataFrame(chord_data_pct, columns=['from', 'to', 'value'])
df_chord_pct.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_percent_GMM_cluster.csv", index=False)


chord_data_median = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue
            if var not in REVERSE_DIRECTION_COLS:
                if np.nanmedian(temp_data[var]) > np.nanmedian(data[var]):
                    value += 1
            else:
                if np.nanmedian(temp_data[var]) < np.nanmedian(data[var]):
                    value += 1

        chord_data_median.append([f'Cluster {c}', system, value])

df_chord_median = pd.DataFrame(chord_data_median, columns=['from', 'to', 'value'])
df_chord_median.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_median_GMM_cluster.csv", index=False)


# # 层次聚类生成和弦数据

# In[2]:


# 定义系统映射（用你的 system_mapping）
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
    'CK-MB': 'Circulation'
}



REVERSE_DIRECTION_COLS = {
    'Albumin',
    'Hemoglobin',
    'Platelet',
    'Red_blood_cell',
    'MAP',
    'PH',
    'P_F',
    'Base_Excess',
}




import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv(r"E:\100-科研\667_总数据版本\007层次聚类\插补后数据_cluster0805.csv")

cluster_col = 'cluster'
C = data[cluster_col].nunique()  # 聚类总数，例如3类或4类

# 构建 system to variable list（和最开始的 LAB_VARS 等价）
system_vars = {}
for var, system in system_mapping.items():
    system_vars.setdefault(system, []).append(var)


chord_data_pct = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]
    Nc = len(temp_data)

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue  # 某些变量可能缺失
            if var not in REVERSE_DIRECTION_COLS:
                n = len(temp_data[temp_data[var] > np.nanmedian(data[var])])
            else:
                n = len(temp_data[temp_data[var] < np.nanmedian(data[var])])
            value += 100 * n / Nc

        chord_data_pct.append([f'Cluster {c}', system, value])

df_chord_pct = pd.DataFrame(chord_data_pct, columns=['from', 'to', 'value'])
df_chord_pct.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_percent_ALG_cluster.csv", index=False)


chord_data_median = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue
            if var not in REVERSE_DIRECTION_COLS:
                if np.nanmedian(temp_data[var]) > np.nanmedian(data[var]):
                    value += 1
            else:
                if np.nanmedian(temp_data[var]) < np.nanmedian(data[var]):
                    value += 1

        chord_data_median.append([f'Cluster {c}', system, value])

df_chord_median = pd.DataFrame(chord_data_median, columns=['from', 'to', 'value'])
df_chord_median.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_median_ALG_cluster.csv", index=False)


# # LPA聚类生成和弦数据
# 

# In[4]:


# 定义系统映射（用你的 system_mapping）
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
    'CK-MB': 'Circulation'
}



REVERSE_DIRECTION_COLS = {
    'Albumin',
    'Hemoglobin',
    'Platelet',
    'Red_blood_cell',
    'MAP',
    'PH',
    'P_F',
    'Base_Excess',
}




import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv(r"E:\100-科研\667_总数据版本\006LPA\插补后数据_LPAcluster0805.csv")

cluster_col = 'Class'
C = data[cluster_col].nunique()  # 聚类总数，例如3类或4类

# 构建 system to variable list（和最开始的 LAB_VARS 等价）
system_vars = {}
for var, system in system_mapping.items():
    system_vars.setdefault(system, []).append(var)


chord_data_pct = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]
    Nc = len(temp_data)

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue  # 某些变量可能缺失
            if var not in REVERSE_DIRECTION_COLS:
                n = len(temp_data[temp_data[var] > np.nanmedian(data[var])])
            else:
                n = len(temp_data[temp_data[var] < np.nanmedian(data[var])])
            value += 100 * n / Nc

        chord_data_pct.append([f'Cluster {c}', system, value])

df_chord_pct = pd.DataFrame(chord_data_pct, columns=['from', 'to', 'value'])
df_chord_pct.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_percent_LPA_cluster.csv", index=False)


chord_data_median = []

for c in range(1, C+1):
    temp_data = data[data[cluster_col] == c]

    for system, var_list in system_vars.items():
        value = 0
        for var in var_list:
            if var not in data.columns:
                continue
            if var not in REVERSE_DIRECTION_COLS:
                if np.nanmedian(temp_data[var]) > np.nanmedian(data[var]):
                    value += 1
            else:
                if np.nanmedian(temp_data[var]) < np.nanmedian(data[var]):
                    value += 1

        chord_data_median.append([f'Cluster {c}', system, value])

df_chord_median = pd.DataFrame(chord_data_median, columns=['from', 'to', 'value'])
df_chord_median.to_csv(r"E:\100-科研\667_总数据版本\008用于做和弦图的数据\chord_lab_median_LPA_cluster.csv", index=False)


# In[ ]:




