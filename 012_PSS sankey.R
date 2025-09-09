rm(list = ls())
gc()
setwd("E:\\100-科研\\667_总数据版本\\014PSS四分类")
library(ggalluvial)
library(dplyr)
library(readr)
library(ggplot2)

# 1) 读取数据
data <- read_csv("E:/100-科研/667_总数据版本/007层次聚类/插补后数据_cluster0805.csv")

# 2) PSS 四分位
data <- data %>%
  mutate(PSS_group = cut(
    PSS,
    breaks = quantile(PSS, probs = seq(0, 1, 0.25), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("< 25%", "25% - 50%", "50% - 75%", "> 75%")
  ))

# 3) cluster → 罗马数字
data <- data %>%
  mutate(cluster_label = recode(as.character(cluster),
                                "1" = "I", "2" = "II", "3" = "III", "4" = "IV"))

# 4) 聚合
df_plot <- data %>%
  count(cluster_label, PSS_group)

# 5) 固定配色（只保留 4 种颜色）
base_colors <- c(
  "I"   = "#F9C77E",
  "II"  = "#7B967A",
  "III" = "#79A3D9",
  "IV"  = "#CE4257"
)

# 6) 绘图
p <- ggplot(df_plot,
            aes(axis1 = cluster_label, axis2 = PSS_group, y = n)) +
  # 流线：颜色跟随 cluster
  geom_alluvium(aes(fill = cluster_label), alpha = 0.7, width = 1/8, color = NA) +
  
  # 节点块：左边 cluster 用纯色，右边 PSS 节点半透明灰，让流线混色可见
  geom_stratum(width = 1/8, color = "black", alpha = 0) +
  
  # 节点文字
  geom_text(stat = "stratum", aes(label = after_stat(stratum)),
            size = 5, family = "Arial", color = "black") +
  
  # 固定流线颜色（只映射 cluster）
  scale_fill_manual(values = base_colors) +
  
  # X 轴
  scale_x_discrete(limits = c("", ""),
                   expand = c(.05, .05)) +
  
  # 主题
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    text = element_text(family = "Arial")
  )

# 显示
print(p)

# 导出 PDF（可在 AI 中编辑）
ggsave("sankey_PSS_cluster_mixednodes.pdf", p, width = 10, height = 8, device = cairo_pdf)
