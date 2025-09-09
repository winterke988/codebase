# Clear the workspace and memory
rm(list = ls())
gc()

# 加载必要的库
# Load necessary packages
# 如果你还没有安装这些包，请先运行 install.packages("...")
# If you haven't installed these packages, run install.packages("...")
library(tidyverse)
library(readr)
library(ggalluvial)
library(Cairo)

# 读取数据
# Read the data from the specified path
df_comparison <- read_csv("E:/100-科研/667_总数据版本/016三种算法对比/三种算法.csv")

# 检查列名以确保与代码匹配
# Check column names to ensure they match the code
print("Column names in the data frame:")
print(colnames(df_comparison))

# --- 构建冲积图的数据 ---
# --- Build data for the Sankey plot ---



# 2. 创建一个用于标签重命名的数据框
# 2. Create a data frame for label renaming
label_map <- data.frame(
  original = c("1", "2", "3", "4"),
  renamed = c("I", "II", "III", "IV")
)

# ---- 方案 1 ----
df_tidy <- df_comparison %>%
  mutate(
    AHC_cluster = as.factor(AHC_cluster),
    GMM_cluster = as.factor(GMM_cluster),
    LPA_cluster = as.factor(LPA_cluster)
  ) %>%
  group_by(AHC_cluster, GMM_cluster, LPA_cluster) %>%
  summarise(freq = n(), .groups = "drop")

sankey_plot1 <- ggplot(
  data = df_tidy,
  aes(axis1 = AHC_cluster, axis2 = GMM_cluster, axis3 = LPA_cluster, y = freq)
) +
  scale_x_discrete(limits = c("AHC", "GMM", "LPA"), expand = c(.1, .1)) +
  geom_alluvium(aes(fill = AHC_cluster), alpha = 0.6) +
  geom_stratum(aes(fill = after_stat(stratum)), color = "black", alpha = 0.6) +
  geom_text(stat = "stratum",
            aes(label = label_map$renamed[match(after_stat(stratum), label_map$original)]),
            size = 4, family = "Arial") +
  labs(y = "Frequency") +
  scale_fill_manual(values = c(
    "1" = "#79A3D9",
    "2" = "#7B967A",
    "3" = "#F9C77E",
    "4" = "#CE4257"
  ), name = "Subphenotype") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.title.x = element_blank(),
    text = element_text(family = "Arial")
  )
sankey_plot1


# 打印并保存图表为 PDF
# Print and save the plot as a PDF

ggsave(
  filename = "sankey_plot.pdf",
  plot = sankey_plot1,
  width = 10,
  height = 8,
  device = cairo_pdf
)






# ---- 方案 1 ----
df_tidy <- df_comparison %>%
  mutate(
    AHC_cluster = as.factor(AHC_cluster),
    GMM_cluster = as.factor(GMM_cluster),
    LPA_cluster = as.factor(LPA_cluster)
  ) %>%
  group_by(AHC_cluster, GMM_cluster, LPA_cluster) %>%
  summarise(freq = n(), .groups = "drop")

sankey_plot2 <- ggplot(
  data = df_tidy,
  aes(axis1 = AHC_cluster, axis2 = GMM_cluster, axis3 = LPA_cluster, y = freq)
) +
  scale_x_discrete(limits = c("AHC", "GMM", "LPA"), expand = c(.1, .1)) +
  geom_alluvium(aes(fill = AHC_cluster), alpha = 0.6) +
  geom_stratum(aes(fill = after_stat(stratum)), color = "black", alpha = 0.6) +
  geom_text(stat = "stratum",
            aes(label = label_map$renamed[match(after_stat(stratum), label_map$original)]),
            size = 4, family = "Arial") +
  labs(y = "Frequency") +
  scale_fill_manual(values = c(
    "1" = "pink",
    "2" = "#82ADD0",
    "3" = "#94C594",
    "4" = "#F1B66D"
  ), name = "Subphenotype") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    axis.title.x = element_blank(),
    text = element_text(family = "Arial")
  )
sankey_plot2


ggsave(
  filename = "sankey_plot2.pdf",
  plot = sankey_plot2,
  width = 10,
  height = 8,
  device = cairo_pdf
)










































