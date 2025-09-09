# 清空环境
rm(list = ls())
gc()
setwd("E:\\100-科研\\667_总数据版本\\012生存分析")

getwd()
library(readr)


library(survival)
library(survminer)
library(dplyr)

# 1. 读取数据
data <- read.csv("E:\\100-科研\\667_总数据版本\\012生存分析\\插补后数据_all_cluster0805.csv")


names(data)
# 加载必要包
library(survival)
library(survminer)

# 1. 创建 Surv 对象（假设你已经有 surv_obj 和 LPA_cluster）
surv_obj <- Surv(time = data$Survival_Time, event = data$X28_day_mortality)
fit <- survfit(surv_obj ~ cluster, data= data)

# 2. 自定义颜色和标签
color_map <- c("#79A3D9", "#7B967A", "#F9C77E", "#CE4257")  # 蓝、绿灰、橙黄、红
legend_labels <- c("I", "II", "III", "IV")  # 罗马数字标签

# 3. 绘图
ggsurvplot(
  fit,
  data = data,
  risk.table = F,                # 风险表
  pval = TRUE,                      # 显示 p 值
  conf.int = FALSE,                 # 不显示置信区间
  legend.title = "Subphenotype",         # 图例标题
  legend.labs = legend_labels,      # 自定义图例标签
  xlab = "Days",
  ylab = "Survival Probability",
  #title = "Kaplan-Meier Survival Curve by Subtype",
  xlim = c(0, 28),                  # 限制横轴到28天
  break.time.by = 7,               # 每7天一个刻度
  risk.table.height = 0.25,        # 风险表高度
  palette = color_map,             # 使用自定义颜色
  surv.plot.height = 0.75,         # 生存曲线高度占比
  ggtheme = theme_minimal(base_size = 14) +  # 更简洁美观的主题
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "top"
    )
)
ggsurvplot(
  fit,
  data = data,
  risk.table = FALSE,
  pval = F,
  conf.int = FALSE,
  legend = "none",                 # 🚫 不显示图例
  xlab = "Days",
  ylab = "Survival Probability",
  xlim = c(0, 28),
  break.time.by = 7,
  risk.table.height = 0.25,
  palette = color_map,
  surv.plot.height = 0.75,
  ggtheme = theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid = element_blank(),               # 去掉所有网格线
      axis.line = element_line(color = "black"),  # 显示坐标轴线
      axis.text.y = element_text(face = "plain")  # y轴刻度文字不加粗
    )
)


# 加载包（确保先加载）
library(survival)
library(survminer)

# 颜色和标签
color_map <- c("#79A3D9", "#7B967A", "#F9C77E", "#CE4257")  # 你自定义的配色
legend_labels <- c("I", "II", "III", "IV")                 # 罗马数字标签
#不加置信区间
# Kaplan-Meier 生存曲线绘制
ggsurvplot(
  fit,
  data = data,
  risk.table = TRUE,                
  pval = TRUE,                      
  conf.int = TRUE,                 
  legend.title = "Subphenotype",    
  legend.labs = legend_labels,      
  xlab = "Days",
  ylab = "Survival Probability",
  title = "",
  xlim = c(0, 28),                  
  break.time.by = 7,               
  risk.table.height = 0.25,        
  palette = color_map,             
  surv.plot.height = 0.75,         
  
  # 美化主题：去掉网格线，实线坐标轴
  ggtheme = theme_minimal(base_size = 14) +
    theme(
      panel.grid = element_blank(),            # 去掉所有网格线
      axis.line = element_line(color = "black"), # 实线坐标轴
      axis.ticks = element_line(color = "black"),
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "top"
    ),
  
  # 让风险表颜色与主图一致
  risk.table.col = "strata"
)














