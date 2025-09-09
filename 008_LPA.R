# 清理工作空间
rm(list = ls())
gc()

# 设置工作目录
setwd("E:\\100-科研\\667_总数据版本\\006LPA")
getwd()

# 加载必要的库
library(readr)
library(mclust)
library(tidyverse)
library(tidyLPA)
library(ggplot2)

# 读取数据（去除 ID）
data <- read.csv("E:/100-科研/667_总数据版本/003对数转换/scaled_for_clustering_with_ID.csv")
data_scaled <- data[, !names(data) %in% c("ID", "id")]

# 重新估计简化模型（支持 LMR / BLRT）
models_lrt <- data_scaled %>%
  single_imputation() %>%
  estimate_profiles(1:9, variances = "equal", covariances = "zero")  # 更简模型结构

# 查看拟合指标
fits_lrt <- get_fit(models_lrt)
print(fits_lrt)

fits_lrt %>%  
  select(Classes, BLRT_val, BLRT_p)

# 输出每个模型中，各类的样本数量（profile 结构）
cat("\n================= 各模型中每类样本数（Profile 分布） =================\n")
for (i in 1:length(models_lrt)) {
  cat(paste0("---- Model ", i, " ----\n"))
  classification <- get_data(models_lrt[[i]])
  print(table(classification$Class))
}

# === 第一步：提取4类模型 ===
model_4class <- models_lrt[[4]]  # 第4个模型，对应 Classes = 4

# === 第二步：重新编码模型内部的 Class 标签 ===
# get_data() 函数返回的是模型结果的一个数据框，我们可以直接修改这个数据框
classified_data <- get_data(model_4class)

# 根据你的需求重新编码 Class 标签
# 原来的1对应现在的1
# 原来的2对应现在的3
# 原来的3对应现在的4
# 原来的4对应现在的2

classified_data$Class <- classified_data$Class %>%
  recode("1" = "1", "2" = "3", "3" = "4", "4" = "2") %>%
  # 将重新编码后的标签转换为因子，以便ggplot正确排序
  factor(levels = c("1", "2", "3", "4"))

# 将修改后的数据重新放回模型对象中，这样 plot_profiles() 就会使用新的标签
model_4class$classified_data <- classified_data

# === 第三步：绘制重新编码后的 Profile 图 ===
# 再次调用 plot_profiles()
# === 第三步：绘制重新编码后的 Profile 图 ===
# 再次调用 plot_profiles()
p_reordered <- plot_profiles(model_4class) +
  labs(
    #title = "Latent Profile Plot: 4-Class Solution (Reordered)",
    x = "Variables",
    y = "Standardized Mean"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    # 设置所有文本的字体为 Arial
    text = element_text(family = "Arial")
  )

# 保存新的图，并指定设备为 cairo_pdf
ggsave(filename = "LPA_4class_profile_plot_reordered.pdf", plot = p_reordered, width = 20, height = 6, device = cairo_pdf)



# === 第四步：绘制你想要的单峰概率分布直方图 ===
cat("\n================= 绘制你想要的单峰直方图 =================\n")

# 获取分类的后验概率（posterior probabilities）
classified_with_probs <- get_data(model_4class)

# 打印数据框的列名，以便你检查概率列的真实名称
print("Columns in classified_with_probs:")
print(colnames(classified_with_probs))

# 将概率数据从宽格式转换为长格式
probs_long <- classified_with_probs %>%
  select(Class, CPROB1, CPROB2, CPROB3, CPROB4) %>%
  pivot_longer(
    cols = starts_with("CPROB"),
    names_to = "Subphenotype",
    values_to = "Probability"
  )

# 重新编码类别标签以匹配之前的重新排序
probs_long$Class_reordered <- probs_long$Class %>%
  recode("1" = "1", "2" = "3", "3" = "4", "4" = "2") %>%
  factor(levels = c("1", "2", "3", "4"))

# 筛选出每个样本概率最高的那个亚型
# Mplus output 的 Class 列已经是最有可能的分类，所以我们只需要根据这个来筛选
final_probs <- probs_long %>%
  # 筛选出每个样本被最高概率分类的那一行
  filter(
    (Class == "1" & Subphenotype == "CPROB1") |
      (Class == "2" & Subphenotype == "CPROB2") |
      (Class == "3" & Subphenotype == "CPROB3") |
      (Class == "4" & Subphenotype == "CPROB4")
  )

# 为每个亚型重新编码标签
final_probs <- final_probs %>%
  mutate(
    Subphenotype_reordered = case_when(
      Subphenotype == "CPROB1" ~ "Subphenotype I",
      Subphenotype == "CPROB2" ~ "Subphenotype II",
      Subphenotype == "CPROB3" ~ "Subphenotype III",
      Subphenotype == "CPROB4" ~ "Subphenotype IV",
      TRUE ~ Subphenotype
    )
  )

# 绘制直方图，为每个亚型使用不同的颜色
p_single_peak_histograms <- ggplot(final_probs, aes(x = Probability)) +
  geom_histogram(
    aes(fill = Subphenotype_reordered), # 根据亚型名称填充颜色
    bins = 10,
    color = "white"
  ) +
  facet_wrap(~ Subphenotype_reordered, scales = "free_y") + # 为每个亚型创建一个子图
  labs(
    title = "LPA Probability of Subphenotype in Derivation cohort",
    x = "Probability",
    y = "Frequency"
  ) +
  # 添加手动颜色比例尺，以便自定义颜色
  scale_fill_manual(values = c(
    "Subphenotype I" = "lightgreen",
    "Subphenotype II" = "lightblue",
    "Subphenotype III" = "orange",
    "Subphenotype IV" = "pink"
  )) +
  theme_minimal() +
  theme(
    text = element_text(family = "Arial"), # 保持 Arial 字体
    strip.text = element_text(face = "bold"), # 子图标题加粗
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# 保存直方图
ggsave(filename = "LPA_4class_single_peak_histograms.pdf", plot = p_single_peak_histograms, width = 10, height = 8, device = cairo_pdf)








# === 第四步：绘制概率分布直方图 ===
cat("\n================= 绘制概率分布直方图 =================\n")

# 获取分类的后验概率（posterior probabilities）
# tidyLPA的get_data()函数通常会包含这些概率
classified_with_probs <- get_data(model_4class)

# 打印数据框的列名，以便你检查概率列的真实名称
print("Columns in classified_with_probs:")
print(colnames(classified_with_probs))

# 重新编码类别标签以匹配之前的重新排序
classified_with_probs$Class_reordered <- classified_with_probs$Class %>%
  recode("1" = "1", "2" = "3", "3" = "4", "4" = "2") %>%
  factor(levels = c("1", "2", "3", "4"))

# 将概率数据从宽格式转换为长格式，以便facet_wrap()使用
# 注意：现在我们根据你提供的列名来指定概率列
probs_long <- classified_with_probs %>%
  select(Class_reordered, CPROB1, CPROB2, CPROB3, CPROB4) %>%
  pivot_longer(
    cols = starts_with("CPROB"),
    names_to = "Subphenotype",
    values_to = "Probability"
  ) %>%
  # 为每个亚型重新编码标签，以匹配你希望的I, II, III, IV
  # 这里的映射根据你的旧类别和新类别顺序进行调整
  mutate(
    Subphenotype_reordered = case_when(
      Subphenotype == "CPROB1" ~ "Subphenotype I",
      Subphenotype == "CPROB2" ~ "Subphenotype II",
      Subphenotype == "CPROB3" ~ "Subphenotype III",
      Subphenotype == "CPROB4" ~ "Subphenotype IV",
      TRUE ~ Subphenotype
    )
  )

# 绘制直方图，只使用X轴的概率数据
p_histograms <- ggplot(probs_long, aes(x = Probability)) +
  geom_histogram(
    fill = "#5DBC96", # 使用单一颜色
    bins = 10,
    color = "white"
  ) +
  facet_wrap(~ Subphenotype_reordered, scales = "free_y") + # 为每个亚型创建一个子图
  labs(
    title = "LPA Probability of Subphenotype in Derivation cohort",
    x = "Probability",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    text = element_text(family = "Arial"), # 保持 Arial 字体
    strip.text = element_text(face = "bold"), # 子图标题加粗
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
p_histograms
# 保存直方图
ggsave(filename = "LPA_4class_histograms_reordered3.pdf", plot = p_histograms, width = 10, height = 8, device = cairo_pdf)



# === 第四步：绘制 BIC 曲线图 ===
cat("\n================= 绘制 BIC 曲线图 =================\n")

# 获取拟合结果数据框
fits <- get_fit(models_lrt)

# 绘制 BIC 曲线
p_bic <- ggplot(fits, aes(x = Classes, y = BIC)) +
  geom_point(color ='#79A3D9',size = 3) +
  geom_line(color = '#79A3D9', size = 1) + # 将线条颜色改为黑色
  # 添加一条虚线来标记最优解（4类）
  geom_vline(xintercept = 4, linetype = "dashed", color = "red") +
  labs(
    #title = "BIC Curve for Latent Profile Analysis",
    x = "Number of Latent Classes",
    y = "BIC"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    text = element_text(family = "Arial"),
    axis.text = element_text(size = 12), # 增大坐标轴数字大小
    axis.line.x = element_line(color = "black"), # 显示x轴线
    axis.line.y = element_line(color = "black")  # 显示y轴线
  ) +
  scale_x_continuous(breaks = 1:9) + # 设置 x 轴刻度为 1-9 的整数
  annotate("text", x = 6, y = max(fits$BIC), label = "Optimal profiles: n = 4", color = "black", size = 5, hjust = 0, vjust = 1)


p_bic

# 保存 BIC 曲线图
ggsave(filename = "LPA_bic_curve.pdf", plot = p_bic, width = 10, height = 8, device = cairo_pdf)




# === 第四步：绘制概率分布直方图 ===
cat("\n================= 绘制概率分布直方图 =================\n")

# 获取分类的后验概率（posterior probabilities）
classified_with_probs <- get_data(model_4class)

# 打印数据框的列名，以便你检查概率列的真实名称
print("Columns in classified_with_probs:")
print(colnames(classified_with_probs))

# 重新编码类别标签以匹配之前的重新排序
classified_with_probs$Class_reordered <- classified_with_probs$Class %>%
  recode("1" = "1", "2" = "3", "3" = "4", "4" = "2") %>%
  factor(levels = c("1", "2", "3", "4"))

# 将概率数据从宽格式转换为长格式，以便facet_wrap()使用
# 注意：现在我们根据你提供的列名来指定概率列
probs_long <- classified_with_probs %>%
  select(Class_reordered, CPROB1, CPROB2, CPROB3, CPROB4) %>%
  pivot_longer(
    cols = starts_with("CPROB"),
    names_to = "Subphenotype",
    values_to = "Probability"
  ) %>%
  # 为每个亚型重新编码标签，以匹配你希望的I, II, III, IV
  # 这里的映射根据你的旧类别和新类别顺序进行调整
  mutate(
    Subphenotype_reordered = case_when(
      Subphenotype == "CPROB1" ~ "Subphenotype I",
      Subphenotype == "CPROB2" ~ "Subphenotype II",
      Subphenotype == "CPROB3" ~ "Subphenotype III",
      Subphenotype == "CPROB4" ~ "Subphenotype IV",
      TRUE ~ Subphenotype
    )
  )

# 绘制直方图，为每个亚型使用不同的颜色
p_histograms <- ggplot(probs_long, aes(x = Probability)) +
  geom_histogram(
    aes(fill = Subphenotype_reordered), # 根据亚型名称填充颜色
    bins = 10,
    color = "white"
  ) +
  facet_wrap(~ Subphenotype_reordered, scales = "free_y") + # 为每个亚型创建一个子图
  labs(
    title = "LPA Probability of Subphenotype in Derivation cohort",
    x = "Probability",
    y = "Frequency"
  ) +
  # 添加手动颜色比例尺，以便自定义颜色
  scale_fill_manual(values = c(
    "Subphenotype I" = "lightgreen",
    "Subphenotype II" = "lightblue",
    "Subphenotype III" = "orange",
    "Subphenotype IV" = "pink"
  )) +
  theme_minimal() +
  theme(
    text = element_text(family = "Arial"), # 保持 Arial 字体
    strip.text = element_text(face = "bold"), # 子图标题加粗
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none" # 移除图例，因为颜色已经通过子图标题表达了
  )
p_histograms
# 保存直方图
ggsave(filename = "LPA_4class_histograms_reordered4.pdf", plot = p_histograms, width = 10, height = 8, device = cairo_pdf)

