# -------------------------------------------------------------------
# 策略一: PSM + Firth logistic + 分层 forest plot
# -------------------------------------------------------------------

rm(list = ls())
gc()

library(tidyverse)
library(MatchIt)
library(cobalt)
library(broom)
library(logistf)  # Firth logistic
library(ggplot2)
library(cowplot)
theme_set(theme_minimal(base_size = 12))

# 1. 读取数据
file_path <- "E:/100-科研/667_总数据版本/017异质性分析/插补后数据_cluster修改名称.csv"
df <- read_csv(file_path)

# 确保变量类型
df <- df %>%
  mutate(
    HVHF = as.factor(HVHF),
    cluster = factor(cluster, levels = c(1,2,3,4)),
    day_28_mortality = as.integer(day_28_mortality)
  )

# 2. PSM
confounders <- c('Age', 'Heart_Rate', 'Neutrophil', 'C_reactive_protein', 'Procalcitonin',
                 'Hemoglobin', 'Platelet', 'BUN', 'INR', 'D_dimer', 'PH',
                 'CK_MB', 'Alanine_aminotransferase', 'White_blood_cell', 'PSS')

psm_formula <- as.formula(paste("HVHF ~", paste(confounders, collapse = " + ")))
match_result <- matchit(psm_formula, data = df, method = "nearest", ratio = 1, caliper = 0.3)
matched_data <- match.data(match_result)


library(tableone)

vars <- confounders
factorVars <- c("HVHF", "cluster")

# 指定正态/非正态变量
normalVars   <- c("Hemoglobin", "Heart_Rate")
nonnormalVars <- setdiff(vars, c(factorVars, normalVars))

# 匹配前
tab_unmatched <- CreateTableOne(
  vars = vars, 
  strata = "HVHF", 
  data = df, 
  factorVars = factorVars, 
  test = FALSE
)
print(tab_unmatched, smd = TRUE, nonnormal = nonnormalVars)

# 匹配后
tab_matched <- CreateTableOne(
  vars = vars, 
  strata = "HVHF", 
  data = matched_data, 
  factorVars = factorVars, 
  test = FALSE
)
print(tab_matched, smd = TRUE, nonnormal = nonnormalVars)


# 转成数据框
tab_unmatched_df <- as.data.frame(print(tab_unmatched, smd = TRUE, nonnormal = nonnormalVars, quote = FALSE, noSpaces = TRUE))
tab_matched_df   <- as.data.frame(print(tab_matched,   smd = TRUE, nonnormal = nonnormalVars, quote = FALSE, noSpaces = TRUE))

# 写出到 CSV
write.csv(tab_unmatched_df, "baseline_unmatched.csv", row.names = TRUE)
write.csv(tab_matched_df,   "baseline_matched.csv",   row.names = TRUE)




# 检查匹配平衡
bal.tab(match_result)
love.plot(match_result, binary = "std", thresholds = c(m = .1))

library(cobalt)
library(ggplot2)

# 检查系统字体（确保有 Arial）
# windows 系统一般自带 Arial
windowsFonts(Arial = windowsFont("Arial"))

# 画 Love plot
p <- love.plot(
  match_result,
  binary = "std",
  thresholds = c(m = .1)
) + theme(text = element_text(family = "Arial"))

# 保存为 PDF (矢量格式, AI 可以直接打开编辑)
ggsave("loveplot.pdf", p, width = 8, height = 6, device = cairo_pdf)

# 也可以保存为 SVG
ggsave("loveplot.svg", p, width = 8, height = 6, device = "svg")






# 3. 整体效应（matched）
overall_model <- glm(day_28_mortality ~ HVHF, data = matched_data, family = binomial)
summary(overall_model)
overall_tidy <- tidy(overall_model, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term == "HVHF1") %>%
  transmute(label = "Overall (matched)", OR = estimate, LCL = conf.low, UCL = conf.high)
print(overall_tidy)

# 4. Firth logistic 交互模型（异质性）
firth_model <- logistf(day_28_mortality ~ HVHF * cluster, data = matched_data)
print(firth_model)
p_interaction <- firth_model$prob[7]  # 交互整体 Wald p
p_interaction_label <- paste0("Firth interaction p = ", signif(p_interaction, 3))

#把结果导出来
firth_results <- data.frame(
  term = names(firth_model$coefficients),
  estimate = firth_model$coefficients,
  OR = exp(firth_model$coefficients),
  LCL = exp(firth_model$ci.lower),
  UCL = exp(firth_model$ci.upper),
  p = firth_model$prob
)

print(firth_results)

# 保存为 CSV / Word
write.csv(firth_results, "firth_full_results.csv", row.names = FALSE)
















# 5. 分层分析（各 cluster）
by_cluster <- matched_data %>%
  group_by(cluster) %>%
  group_modify(~ {
    m <- glm(day_28_mortality ~ HVHF, data = .x, family = binomial)
    tidy(m, conf.int = TRUE, exponentiate = TRUE) %>%
      filter(term == "HVHF1") %>%
      transmute(OR = estimate, LCL = conf.low, UCL = conf.high)
  }) %>%
  ungroup() %>%
  mutate(label = paste0("Cluster ", cluster)) %>%
  select(label, OR, LCL, UCL)

print(by_cluster)


by_cluster <- matched_data %>%
  group_by(cluster) %>%
  group_map(~{
    data_sub <- .x
    # 如果 HVHF 在这个 cluster 内只有一个水平
    if(length(unique(data_sub$HVHF)) < 2){
      tibble(OR = NA_real_, LCL = NA_real_, UCL = NA_real_, label = unique(.y$cluster))
    } else {
      m <- glm(day_28_mortality ~ HVHF, data = data_sub, family = binomial)
      tdy <- tidy(m, conf.int = TRUE, exponentiate = TRUE)
      tdy_hv <- tdy %>% filter(term == "HVHF1")
      if(nrow(tdy_hv) == 0){
        tibble(OR = NA_real_, LCL = NA_real_, UCL = NA_real_, label = unique(.y$cluster))
      } else {
        tibble(OR = tdy_hv$estimate, LCL = tdy_hv$conf.low, UCL = tdy_hv$conf.high,
               label = unique(.y$cluster))
      }
    }
  }, .keep = TRUE) %>% 
  bind_rows()

by_cluster <- matched_data %>%
  # Remove rows with NA clusters to avoid issues
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  # Use summarise to create a single row per group
  summarise(
    # Get unique cluster label
    label = first(cluster),
    # If there's only one level of HVHF, return NA. Otherwise, run the model.
    # The `nest()` and `map()` pattern is useful here.
    tidy_glm = list(
      if (length(unique(HVHF)) < 2) {
        tibble(term = "HVHF1", estimate = NA_real_, conf.low = NA_real_, conf.high = NA_real_)
      } else {
        model <- glm(day_28_mortality ~ HVHF, data = cur_data(), family = binomial)
        tidy(model, conf.int = TRUE, exponentiate = TRUE)
      }
    )
  ) %>%
  # Unnest the results and select the row you need
  unnest(tidy_glm) %>%
  filter(term == "HVHF1") %>%
  # 将 select() 改为 dplyr::select()
  dplyr::select(OR = estimate, LCL = conf.low, UCL = conf.high, label)




# Corrected by_cluster code
by_cluster <- matched_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarise(
    # Get unique cluster label and format it to match the counts table
    label = paste0("Cluster ", first(cluster)),
    tidy_glm = list(
      if (length(unique(HVHF)) < 2) {
        tibble(term = "HVHF1", estimate = NA_real_, conf.low = NA_real_, conf.high = NA_real_)
      } else {
        model <- glm(day_28_mortality ~ HVHF, data = cur_data(), family = binomial)
        tidy(model, conf.int = TRUE, exponentiate = TRUE)
      }
    )
  ) %>%
  unnest(tidy_glm) %>%
  filter(term == "HVHF1") %>%
  dplyr::select(OR = estimate, LCL = conf.low, UCL = conf.high, label)

# Now, the rest of your code should work as intended.
# ... (rest of your code for count_overall, count_clusters, forest_df, and ggplot)



















print(by_cluster)
# 6. 事件数 / 总数
count_overall <- matched_data %>%
  summarise(events = sum(day_28_mortality), total = n()) %>%
  mutate(label = "Overall (matched)")

count_clusters <- matched_data %>%
  group_by(cluster) %>%
  summarise(events = sum(day_28_mortality), total = n(), .groups = "drop") %>%
  mutate(label = paste0("Cluster ", cluster))

counts <- bind_rows(count_overall, count_clusters) %>%
  mutate(n_label = paste0("n = ", events, "/", total))

# 7. 合并表格
forest_df <- bind_rows(overall_tidy, by_cluster) %>%
  left_join(counts, by = "label") %>%
  mutate(
    label = factor(label, levels = c("Overall (matched)", paste0("Cluster ", 1:4))),
    or_ci = paste0(sprintf("%.2f", OR), " (", sprintf("%.2f", LCL), ", ", sprintf("%.2f", UCL), ")")
  ) %>%
  arrange(desc(label)) %>%
  mutate(y = row_number())

# 8. forest plot

library(ggplot2)

# 假设 forest_df 已经准备好
x_limits <- c(0.01, 10)
x_breaks <- c(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)

p_forest <- ggplot(forest_df, aes(x = OR, y = y)) +
  # 中心参考线
  geom_vline(xintercept = 1, linetype = "dashed") +
  # 置信区间横线
  geom_errorbarh(aes(xmin = LCL, xmax = UCL), height = 0.15) +
  # OR 点
  geom_point(size = 3) +
  # log10 坐标
  scale_x_log10(limits = x_limits, breaks = x_breaks) +
  scale_y_continuous(breaks = forest_df$y, labels = forest_df$label) +
  
  # 左侧人数列（不带横线，用 geom_text）
  geom_text(aes(x = 0.01, label = n_label), hjust = 0, size = 4, family = "Arial") +
  
  # 右侧 CI 列（同样用 geom_text，空白背景）
  geom_text(aes(x = 10, label = or_ci), hjust = 1, size = 4, family = "Arial") +
  
  labs(#title = "Effect of HVHF on 28-day Mortality (Matched Cohort)",
       subtitle = p_interaction_label,
       x = "Odds Ratio (log scale)", y = NULL) +
  
  # 主题美化
  theme_classic(base_family = "Arial") +   # Arial 字体，去掉背景
  theme(
    plot.title = element_text(face = "bold", size = 16, family = "Arial"),
    plot.subtitle = element_text(size = 14, family = "Arial"),
    axis.text.y = element_text(hjust = 0, size = 13, family = "Arial"),
    axis.text.x = element_text(size = 12, family = "Arial"),
    axis.title.x = element_text(size = 14, family = "Arial"),
    axis.line.y = element_blank(),   # 去掉 y 轴线
    axis.ticks.y = element_blank()   # 去掉 y 轴刻度
  )

p_forest

ggsave("forest_plot.pdf", p_forest, 
       width = 12, height = 6, units = "in", 
       device = cairo_pdf)  # 用 cairo_pdf 保留可编辑文字





# 基础模型（无交互）
firth_base <- logistf(day_28_mortality ~ HVHF + cluster, data = matched_data)

# 含交互模型
firth_int <- logistf(day_28_mortality ~ HVHF * cluster, data = matched_data)

# 提取 penalized log-likelihood
logL_base <- firth_base$loglik[2]  # 第二个元素是最大似然值
logL_int  <- firth_int$loglik[2]

# 计算 LRT 统计量
LRT_stat <- 2 * (logL_int - logL_base)

# 自由度 = 参数差
df <- length(firth_int$coefficients) - length(firth_base$coefficients)

# 对应 p 值
p_LRT <- 1 - pchisq(LRT_stat, df)

# 输出
LRT_stat
df
p_LRT




# 生成基线表
library(tableone)

vars <- confounders
factorVars <- c("HVHF", "cluster")

# 匹配前
tab_unmatched <- CreateTableOne(
  vars = vars, 
  strata = "HVHF", 
  data = as.data.frame(df),   # 转换 tibble → data.frame
  factorVars = factorVars, 
  test = FALSE
)
print(tab_unmatched, smd = TRUE)

# 匹配后
tab_matched <- CreateTableOne(
  vars = vars, 
  strata = "HVHF", 
  data = as.data.frame(matched_data), 
  factorVars = factorVars, 
  test = FALSE
)
print(tab_matched, smd = TRUE)
colnames(df)


# 可以保存为 word / excel 表格
# library(officer)
# library(flextable)
# ft <- as_flextable(tab_matched)
# save_as_docx(ft, path = "baseline_table.docx")
