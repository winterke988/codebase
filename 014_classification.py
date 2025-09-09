#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 1. 读取数据
data = pd.read_csv(r"E:\\100-科研\\667_总数据版本\\007层次聚类\\插补后数据_cluster0805.csv")

# 2. 特征和标签
TARGET_LABs = [
    'Age', 'Heart_Rate', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
    'Hemoglobin', 'Platelet', 'BUN','INR', 'D_dimer', 'PH',
    'CK-MB', 'Alanine_aminotransferase', 'White_blood_cell','PSS'
]
X = data[TARGET_LABs]
y = data['cluster']

# 3. 重新编码标签，确保从 0 开始（避免报错）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. 构建模型
model = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# 5. 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')

print(f"交叉验证每折准确率: {cv_scores}")
print(f"平均准确率: {np.mean(cv_scores):.3f}")


# In[5]:


from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# 设置全局字体为 Arial（保证 Illustrator 可编辑文字）
import matplotlib as mpl
import matplotlib.pyplot as plt

# 让 Illustrator 可编辑文字
mpl.rcParams['pdf.fonttype'] = 42  # 嵌入 TrueType
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14       # 全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12 # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12 # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12 # 图例字体大小


# 1. 使用 OneVsRestClassifier 包装你的模型
ovr_model = OneVsRestClassifier(model)

# 2. 交叉验证预测概率（y_encoded是一维标签，符合要求）
y_score = cross_val_predict(ovr_model, X, y_encoded, cv=skf, method='predict_proba')

# 3. 二值化标签（计算多分类 ROC 需要）
y_bin = label_binarize(y_encoded, classes=[0,1,2,3])
n_classes = y_bin.shape[1]

# 4. 计算每个类别的 FPR、TPR 和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 5. 计算 micro-average ROC 曲线和 AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 6. 计算 macro-average ROC 曲线和 AUC
# 先聚合所有类别的 FPR 点
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# 对每个类别插值 TPR
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# 求平均
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 7. 绘制 ROC 曲线图
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red', 'orange']

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label=f'Subphenotype {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()

# 保存为 PDF（Illustrator 可编辑文字）
plt.savefig(r"E:\100-科研\667_总数据版本\013分类器\multiclass_ROC.pdf", 
            format="pdf",
           bbox_inches="tight",
           dpi=600)

plt.show()


# In[6]:


from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial
import matplotlib as mpl
import matplotlib.pyplot as plt

# 让 Illustrator 可编辑文字
mpl.rcParams['pdf.fonttype'] = 42  # 嵌入 TrueType
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14       # 全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12 # x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12 # y 轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12 # 图例字体大小

# 预测概率（OvR + CV）
ovr_model = OneVsRestClassifier(model)
y_score = cross_val_predict(ovr_model, X, y_encoded, cv=skf, method='predict_proba')

# 标签二值化
y_bin = label_binarize(y_encoded, classes=[0,1,2,3])
n_classes = y_bin.shape[1]

precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])

# micro-average
precision["micro"], recall["micro"], _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_bin, y_score, average="micro")

# 绘图
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red', 'orange']

for i in range(n_classes):
    plt.plot(recall[i], precision[i], color=colors[i],
             label=f'Subphenotype {le.inverse_transform([i])[0]} (AP = {average_precision[i]:.2f})')

plt.plot(recall["micro"], precision["micro"],
         label=f'micro-average PR (AP = {average_precision["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall Curve (One-vs-Rest)')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# 保存为 PDF（Illustrator 可编辑文字）

plt.savefig(r"E:\100-科研\667_总数据版本\013分类器\multiclass_PR.pdf", 
            format="pdf",
           bbox_inches="tight",
           dpi=600)
plt.show()


# In[4]:


import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection
import matplotlib.pyplot as plt

# 设置全局字体为 Arial
import matplotlib as mpl
import matplotlib.pyplot as plt

# 让 Illustrator 可编辑文字
mpl.rcParams['pdf.fonttype'] = 42  # 嵌入 TrueType
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'

import os

# 目标文件夹
outdir = r"E:\100-科研\667_总数据版本\013分类器"



# -------- 1. 拟合模型 & 计算 SHAP --------
model.fit(X, y_encoded)
explainer = shap.TreeExplainer(model)
_raw = explainer.shap_values(X)

if isinstance(_raw, list):
    shap_class_values = _raw
else:
    shap_class_values = [_raw[:, :, i] for i in range(_raw.shape[2])]

# -------- 2. 特征分组 & 颜色映射（你自己的配置）--------
feature_categories = {
    "Inflammatory markers": ["C_reactive_protein", "Procalcitonin", "Neutrophil"],
    "Hepatic markers": ["Alanine_aminotransferase"],
    "Renal markers": ["BUN"],
    "Hematologic markers": ["Hemoglobin", "Platelet", "White_blood_cell"],
    "Circulation markers": ["Heart_Rate", "CK-MB"],
    "Coagulation markers": ["INR", "D_dimer"],
    "Metabolic markers": ["PH"],
    "General": ["Age", "PSS"]
}
category_order = [
    "Inflammatory markers","Hepatic markers","Renal markers","Hematologic markers",
    "Circulation markers","Coagulation markers","Metabolic markers","General"
]
category_colors = {
    "Inflammatory markers": "#8FB4DC",
    "Hepatic markers": "#FFDD8E",
    "Renal markers": "#70CDBE",
    "Hematologic markers": "#AC99D2",
    "Circulation markers": "#7AC3DF",
    "Coagulation markers": "#EB7E60",
    "Metabolic markers": "#F9C192",
    "General": "#2B879E"
}

def color_of_feature(feat):
    for cat, feats in feature_categories.items():
        if feat in feats:
            return category_colors.get(cat, "#808080")
    return "#808080"

# -------- 3. 固定变量顺序并重排 X/shap_values --------
ordered_features = []
for cat in category_order:
    for f in feature_categories.get(cat, []):
        if f in X.columns:
            ordered_features.append(f)
# 把 X 中未列入的变量放到末尾并归 General
for f in X.columns:
    if f not in ordered_features:
        ordered_features.append(f)
        feature_categories.setdefault("General", []).append(f)

col_index_order = [X.columns.get_loc(f) for f in ordered_features]
X_fixed = X[ordered_features].copy()
shap_class_values_fixed = [sv[:, col_index_order] for sv in shap_class_values]

# 统一 x 轴范围（便于比较）
global_absmax = max(float(np.nanmax(np.abs(sv))) for sv in shap_class_values_fixed)
global_absmax = float(np.ceil(global_absmax * 10) / 10.0)

# 点的渐变 colormap（feature value）
cmap = plt.get_cmap("RdYlBu_r")

# -------- 4. 绘制（每个子型一张图，beeswarm/dot + 统一 colorbar + 左侧方块）--------
for i, class_name in enumerate(le.classes_):
    plt.figure(figsize=(12, 9))

    # 使用 dot/beeswarm 来表现彩色点（点颜色由 feature value 决定）
    shap.summary_plot(
        shap_class_values_fixed[i],
        X_fixed,
        feature_names=ordered_features,
        sort=False,
        max_display=len(ordered_features),
        show=False,
        plot_type="dot"   # beeswarm：点会形成类似“violin”的密度效果
    )

    fig = plt.gcf()
    # 找到“主绘图区”——通过 collections 数量来识别（通常主图 collection 最多）
    axes = fig.get_axes()
    main_ax = max(axes, key=lambda a: len(a.collections))

    # 删除非主轴（也就是 shap 自动画的 colorbar 等），避免出现重复 colorbar
    for ax in axes:
        if ax is not main_ax:
            try:
                fig.delaxes(ax)
            except Exception:
                pass

    # 将主轴的点集合（PathCollection）设置为我们想要的 cmap
    for coll in main_ax.collections:
        if isinstance(coll, PathCollection):
            # 如果有 array（shap 会有），使用其范围设置 color limits
            try:
                arr = coll.get_array()
                if arr is not None:
                    coll.set_cmap(cmap)
                    # 设定 color limits（有些版本需要）
                    coll.set_clim(np.nanmin(arr), np.nanmax(arr))
            except Exception:
                # 兼容性：尝试只设置 cmap
                try:
                    coll.set_cmap(cmap)
                except Exception:
                    pass

    # 有些 shap 版本还需要对 figure 内可 set_cmap 的对象强制设置 cmap
    for fc in fig.get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                try:
                    fcc.set_cmap(cmap)
                except Exception:
                    pass

    # 标题与 x 轴范围
    main_ax.set_title(f"Rest vs Subphenotype {class_name}", fontsize=18, pad=12)
    main_ax.set_xlim(-global_absmax, global_absmax)

    # 在变量名左侧画彩色小方块（精确对应名称）
    labels = [lab.get_text() for lab in main_ax.get_yticklabels()]
    yticklocs = main_ax.get_yticks()
    x_for_square = -0.07   # 往左一点（可调）
    for y, lab in zip(yticklocs, labels):
        col = color_of_feature(lab)
        main_ax.scatter(
            x_for_square, y,
            marker="s", s=90, color=col,
            transform=main_ax.get_yaxis_transform(),
            clip_on=False, zorder=3
        )

    # 底部图例（器官系统）
    legend_handles = [
        Line2D([0], [0], marker="s", color="w",
               label=cat, markerfacecolor=category_colors[cat], markersize=10)
        for cat in category_order if any(f in X.columns for f in feature_categories.get(cat, []))
    ]
    fig.legend(
        handles=legend_handles,
        title="Variable categories",
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02)
    )

    # 使用主轴中的第一个 PathCollection 创建 colorbar（确保只有一个）
    # 找到一个有 array 的 PathCollection
    pcoll_with_array = None
    for coll in main_ax.collections:
        if isinstance(coll, PathCollection):
            if coll.get_array() is not None:
                pcoll_with_array = coll
                break

    if pcoll_with_array is not None:
        cbar = fig.colorbar(pcoll_with_array, ax=main_ax, orientation="vertical", pad=0.02)
        cbar.set_label("Feature value", fontsize=12)
        # 你可自定义 ticks/labels，例如把上端标 High，下端标 Low：
        cbar.ax.set_yticklabels(["Low", "", "High"])  # 或其它自定义（根据需要调整）
    else:
        # 保险 fallback：用一个 ScalarMappable 来创建 colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=main_ax, orientation="vertical", pad=0.02)
        cbar.set_label("Feature value", fontsize=12)

    plt.subplots_adjust(left=0.32, bottom=0.20, right=0.92)
    safe_name = str(class_name).replace("/", "_")
    # 确保文件夹存在
    os.makedirs(outdir, exist_ok=True)

    safe_name = str(class_name).replace("/", "_")

    plt.savefig(os.path.join(outdir, f"shap_subphenotype_{safe_name}_beeswarm_redblue.png"),
                dpi=300, bbox_inches="tight")

    plt.savefig(os.path.join(outdir, f"shap_subphenotype_{safe_name}_beeswarm_redblue.pdf"),
                format="pdf", bbox_inches="tight")
    plt.show()


# In[ ]:




