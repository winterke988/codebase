# ----------------------------
# 次要结局全面分析脚本 （R）
# 假设：文件路径和变量名按用户提供
# 文件: "E:/100-科研/667_总数据版本/018次要结局的分析/次要结局分析.csv"
# 请在运行前检查并根据需要调整协变量列表与 time/status 构造部分
# ----------------------------
rm(list = ls())
gc()
# ---- 0. 安装/加载包（如已安装可跳过安装行） ----
required <- c("readr","dplyr","magrittr","ggplot2","pscl","MASS","broom","tableone","cmprsk","survival","ggpubr")
install_if_missing <- function(pkgs){
  for(p in pkgs) if(!requireNamespace(p,quietly=TRUE)) install.packages(p)
}
install_if_missing(required)

library(readr); library(dplyr); library(magrittr)
library(ggplot2); library(pscl); library(MASS)
library(broom); library(tableone); library(cmprsk); library(survival); library(ggpubr)

# ---- 1. 读入数据并清洗列名 ----
fpath <-"E:\\100-科研\\667_总数据版本\\018次要结局的分析\\002换混淆因素版本\\插补后数据_cluster修改名称2.csv"
df <- read_csv(fpath, show_col_types = FALSE)

# 统一列名：去首尾空格并替换空格为下划线
names(df) <- names(df) %>%
  trimws() %>%
  gsub("[[:space:]]+", "_", .)

# 查看关键变量是否存在
needed_vars <- c("id","Age","Heart_Rate","PH","White_blood_cell","Neutrophil",
                 "C_reactive_protein","Procalcitonin","Hemoglobin","Platelet",
                 "INR","D_dimer","CK_MB","BUN","Alanine_aminotransferase",
                 "Mv_Time","PSS","ICU_LOS","LOS","ICU_FDS","Mv_FDS",
                 "Survival_Time","day_28_mortality","cluster")
miss <- setdiff(needed_vars, names(df))
if(length(miss)>0){
  message("警告：下列变量在数据中未找到，请确认列名（大小写/空格）：\n", paste(miss, collapse=", "))
}

# 确保必要变量类型
df <- df %>%
  mutate(
    cluster = as.factor(cluster),
    day_28_mortality = as.integer(day_28_mortality),
    ICU_FDS = as.numeric(ICU_FDS),
    Mv_FDS = as.numeric(Mv_FDS),
    ICU_LOS = as.numeric(ICU_LOS),
    LOS = as.numeric(LOS),
    Survival_Time = as.numeric(Survival_Time)
  )



# 检查数据结构与列名
head(df)
names(df)
str(df)

# 检查 cluster 列是否存在并查看类别
if("cluster" %in% names(df)) {
  table(df$cluster, useNA="ifany")
} else {
  stop("没有发现 'cluster' 列，请确认列名是否正确（注意空格/大小写）")
}

# ---- 2. 基本描述性：中位数(IQR) + 0 的比例 ----
# 安装并加载
if(!requireNamespace("dplyr", quietly=TRUE)) install.packages("dplyr")
if(!requireNamespace("magrittr", quietly=TRUE)) install.packages("magrittr")
library(dplyr)
library(magrittr)

# 确保 cluster 是因子（可选）
df$cluster <- as.factor(df$cluster)

# 汇总（运行）
desc_ICU <- df %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    ICU_FDS_median = median(ICU_FDS, na.rm=TRUE),
    ICU_FDS_Q1 = quantile(ICU_FDS, 0.25, na.rm=TRUE),
    ICU_FDS_Q3 = quantile(ICU_FDS, 0.75, na.rm=TRUE),
    percent_zero_ICU_FDS = mean(ICU_FDS == 0, na.rm=TRUE) * 100,
    VFDs_median = median(Mv_FDS, na.rm=TRUE),
    VFDs_Q1 = quantile(Mv_FDS, 0.25, na.rm=TRUE),
    VFDs_Q3 = quantile(Mv_FDS, 0.75, na.rm=TRUE),
    percent_zero_VFDs = mean(Mv_FDS == 0, na.rm=TRUE) * 100,
    .groups = "drop"
  ) %>%
  arrange(cluster)

print(desc_ICU)


# 写出描述性表到 CSV
write.csv(desc_ICU, file = "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/ICU_VFDs_descriptive_by_cluster.csv", row.names = FALSE)

# ---- 3. 绘图：0 vs >0 堆积条形 & 箱线 + 抖动 ----
# 每次新开 R 会话都需要加载
library(ggplot2)
# 0 vs >0 比例图 (ICU_FDS)
plot_df1 <- df %>%
  mutate(ICU_zero = ifelse(ICU_FDS==0, "zero", "positive")) %>%
  group_by(cluster, ICU_zero) %>%
  tally() %>%
  group_by(cluster) %>%
  mutate(pct = n / sum(n) * 100)

p1 <- ggplot(plot_df1, aes(x = cluster, y = pct, fill = ICU_zero)) +
  geom_col(position = "stack") +
  labs(title = "ICU_FDS: % zero vs >0 by cluster", y="Percent", x="Cluster") +
  theme_minimal()
p1
ggsave("E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/ICU_FDS_zero_pct_by_cluster.png", p1, width=7, height=5)

# 箱线图 + 抖动（>0 及全部）
p2 <- ggplot(df, aes(x = cluster, y = ICU_FDS)) +
  geom_jitter(width = 0.2, alpha = 0.4, size=1) +
  geom_boxplot(alpha = 0.2, outlier.shape = NA) +
  labs(title = "ICU_FDS distribution by cluster", y="ICU_FDS (days)", x="Cluster") +
  theme_minimal()
p2
ggsave("E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/ICU_FDS_boxplot.png", p2, width=7, height=5)

# 同理为 Mv_FDS
plot_df2 <- df %>%
  mutate(MV_zero = ifelse(Mv_FDS==0, "zero", "positive")) %>%
  group_by(cluster, MV_zero) %>%
  tally() %>%
  group_by(cluster) %>%
  mutate(pct = n / sum(n) * 100)

p3 <- ggplot(plot_df2, aes(x = cluster, y = pct, fill = MV_zero)) +
  geom_col(position = "stack") +
  labs(title = "Mv_FDS: % zero vs >0 by cluster", y="Percent", x="Cluster") +
  theme_minimal()
p3
ggsave("E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/Mv_FDS_zero_pct_by_cluster.png", p3, width=7, height=5)

p4 <- ggplot(df, aes(x = cluster, y = Mv_FDS)) +
  geom_jitter(width = 0.2, alpha = 0.4, size=1) +
  geom_boxplot(alpha = 0.2, outlier.shape = NA) +
  labs(title = "Mv_FDS distribution by cluster", y="Mv_FDS (days)", x="Cluster") +
  theme_minimal()
p4
ggsave("E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/Mv_FDS_boxplot.png", p4, width=7, height=5)

# ---- 4. 组间比较（Kruskal-Wallis + pairwise Wilcoxon） ----
kruskal_ICU <- kruskal.test(ICU_FDS ~ cluster, data = df)
kruskal_MV <- kruskal.test(Mv_FDS ~ cluster, data = df)

pairwise_ICU <- pairwise.wilcox.test(df$ICU_FDS, df$cluster, p.adjust.method = "BH")
pairwise_MV <- pairwise.wilcox.test(df$Mv_FDS, df$cluster, p.adjust.method = "BH")

capture.output(kruskal_ICU, file = "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/kruskal_ICU.txt")
capture.output(kruskal_MV, file = "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/kruskal_MV.txt")
write.csv(as.data.frame(pairwise_ICU$p.value), "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/pairwise_ICU_pvalues.csv")
write.csv(as.data.frame(pairwise_MV$p.value), "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/pairwise_MV_pvalues.csv")








# ===============================
# 次要结局 Two-part (logistic + positive-part) + Fine-Gray 脚本（已修改）
# 说明：针对 ICU_FDS / Mv_FDS 为“非负连续（含小数）且大量0”这一特点
#      正值部分优先使用 Gamma (log link)，若为整数可尝试 glm.nb。
# 输出：模型 summary/csv 保存到 E:/100-科研/667_总数据版本/018次要结局的分析/
# ===============================

outdir <- "E:/100-科研/667_总数据版本/018次要结局的分析/002换混淆因素版本/"

# 必要包（如未安装请先 install.packages(...)）
library(readr)
library(dplyr)
library(magrittr)
library(MASS)      # glm.nb fallback（若需要）
library(broom)     # tidy()
library(survival)  # Surv
library(cmprsk)    # crr

# 如果 df 尚未读入则读入（防错）
if(!exists("df")) {
  df <- readr::read_csv(fpath, show_col_types = FALSE)
  # 规范列名（去空格 -> 下划线）
  names(df) <- names(df) %>% trimws() %>% gsub("[[:space:]]+", "_", .)
}

# 确保 cluster 为因子
df$cluster <- as.factor(df$cluster)

# ---- 1. 定义协变量 ----
# ---- 1. 定义协变量 ----
confounders <- c('Epinephrine', 'HVHF', 'Hemoglobin',
                  'Nutrition_Method',
                 'PSS', 'Age')

# 确保二分类变量转为 factor
df <- df %>%
  mutate(
    Epinephrine = factor(Epinephrine, levels = c(0,1)),
    HVHF = factor(HVHF, levels = c(0,1)),
    Nutrition_Method = factor(Nutrition_Method) # 如果本身就是多分类，就直接 factor()
  )

# 检查哪些协变量在数据中缺失（提醒但不中断）
missing_covs <- setdiff(confounders, names(df))
if(length(missing_covs) > 0) {
  warning("下列协变量未在数据中找到，请核对列名： ", paste(missing_covs, collapse = ", "))
}


# ---- 2. 构造二元零指示变量 ----
df <- df %>%
  mutate(
    ICU_zero = ifelse(is.na(ICU_FDS), NA, ifelse(ICU_FDS == 0, 1, 0)),
    MV_zero  = ifelse(is.na(Mv_FDS),  NA, ifelse(Mv_FDS == 0,  1, 0))
  )

# ---- 3. ZERO 部分（Logistic：是否为 0） ----
zero_formula_str <- paste("ICU_zero ~ cluster +", paste(confounders, collapse = " + "))
zero_formula <- as.formula(zero_formula_str)

glm_zero <- tryCatch(
  glm(zero_formula, data = df, family = binomial(link = "logit"), na.action = na.exclude),
  error = function(e) { message("ICU_zero logistic failed: ", e$message); NULL }
)

capture.output({
  cat("==== ICU_zero logistic summary ====\n")
  if(is.null(glm_zero)) cat("模型拟合失败\n") else print(summary(glm_zero))
}, file = file.path(outdir, "ICU_zero_logistic_summary.txt"))

if(!is.null(glm_zero)) write.csv(broom::tidy(glm_zero), file.path(outdir, "ICU_zero_logistic_tidy.csv"), row.names = FALSE)

# ---- 4. POSITIVE 部分（ICU_FDS > 0） ----
df_pos <- df %>% filter(!is.na(ICU_FDS) & ICU_FDS > 0)

pos_formula_str <- paste("ICU_FDS ~ cluster +", paste(confounders, collapse = " + "))
pos_formula <- as.formula(pos_formula_str)

# 辅助函数：判断是否几乎为整数（处理浮点误差）
is_integer_like <- function(x, tol = 1e-8) {
  x <- x[!is.na(x)]
  if(length(x) == 0) return(FALSE)
  all(abs(x - round(x)) < tol)
}

pos_model_ICU <- NULL
if(nrow(df_pos) == 0) {
  message("没有 ICU_FDS > 0 的样本，跳过正值部分建模。")
} else {
  if(is_integer_like(df_pos$ICU_FDS)) {
    # 如果几乎全为整数：尝试 NB
    pos_model_ICU <- tryCatch(
      glm.nb(pos_formula, data = df_pos),
      error = function(e) {
        message("glm.nb 失败，改用 lm：", e$message)
        lm(pos_formula, data = df_pos)
      }
    )
  } else {
    # 连续正变量 —— 优先用 Gamma (log link)
    pos_model_ICU <- tryCatch(
      glm(pos_formula, data = df_pos, family = Gamma(link = "log")),
      error = function(e) {
        message("Gamma glm 失败，尝试 log-transform lm：", e$message)
        # fallback: log-transform linear model
        lm(as.formula(paste("log(ICU_FDS) ~ cluster +", paste(confounders, collapse = " + "))),
           data = df_pos)
      }
    )
  }
  # 保存 summary 与 tidy（并对 log-link 的结果额外给出指数化估计）
  capture.output({
    cat("==== ICU_FDS positive-part model summary ====\n")
    print(summary(pos_model_ICU))
  }, file = file.path(outdir, "ICU_pos_model_summary.txt"))
  
  # tidy 保存（若 broom 支持该模型类型）
  safe_tidy <- tryCatch(broom::tidy(pos_model_ICU), error = function(e) NULL)
  if(!is.null(safe_tidy)) write.csv(safe_tidy, file.path(outdir, "ICU_pos_model_tidy.csv"), row.names = FALSE)
  
  # 如果是 glm with log-link (Gamma) 或 lm(log(y)~...), 输出 exponentiated effects（近似倍数效应）
  fam <- if(inherits(pos_model_ICU, "glm")) pos_model_ICU$family else NULL
  if(!is.null(fam) && tolower(fam$family) == "gamma" && tolower(fam$link) == "log") {
    coef_tab <- broom::tidy(pos_model_ICU)
    coef_tab <- coef_tab %>%
      mutate(
        exp_est = exp(estimate),
        exp_low = exp(estimate - 1.96 * std.error),
        exp_high = exp(estimate + 1.96 * std.error)
      )
    write.csv(coef_tab, file.path(outdir, "ICU_pos_model_exponentiated_effects.csv"), row.names = FALSE)
  } else if(inherits(pos_model_ICU, "lm") && grepl("^log\\(", deparse(pos_model_ICU$call$formula)[2])) {
    # lm(log(ICU_FDS) ~ ...) 的情形，也可给出 exponentiated解释
    coef_tab <- broom::tidy(pos_model_ICU)
    coef_tab <- coef_tab %>%
      mutate(
        exp_est = exp(estimate),
        exp_low = exp(estimate - 1.96 * std.error),
        exp_high = exp(estimate + 1.96 * std.error)
      )
    write.csv(coef_tab, file.path(outdir, "ICU_pos_model_loglm_exponentiated_effects.csv"), row.names = FALSE)
  }
}

# ---- 5. （可选）hurdle/zeroinfl 注释说明（通常不适用于连续 FDS） ----
# 如果你的 FDS 真的是整数计数（0,1,2,...）且想使用 pscl::hurdle 或 zeroinfl，请在此处启用：
# library(pscl)
# hurdle_icufds <- hurdle(ICU_FDS ~ cluster + ..., data = df, dist = "negbin")
# zeroinfl_icufds <- zeroinfl(ICU_FDS ~ cluster + ... | cluster + ..., data = df, dist = "negbin")
# 注：当前脚本中已默认不使用这些模型，因为 ICU_FDS / Mv_FDS 中存在小数。

# ---- 6. 对 Mv_FDS 做同样的 two-part 分析 ----
# ZERO 部分（logistic）
mv_zero_formula <- as.formula(paste("MV_zero ~ cluster +", paste(confounders, collapse = " + ")))
glm_zero_mv <- tryCatch(glm(mv_zero_formula, data = df, family = binomial(link = "logit"), na.action = na.exclude),
                        error = function(e) { message("MV_zero logistic failed: ", e$message); NULL })
capture.output({
  cat("==== MV_zero logistic summary ====\n")
  if(is.null(glm_zero_mv)) cat("模型拟合失败\n") else print(summary(glm_zero_mv))
}, file = file.path(outdir, "MV_zero_logistic_summary.txt"))
if(!is.null(glm_zero_mv)) write.csv(broom::tidy(glm_zero_mv), file.path(outdir, "MV_zero_logistic_tidy.csv"), row.names = FALSE)

# POSITIVE 部分（Mv_FDS > 0）
df_pos_mv <- df %>% filter(!is.na(Mv_FDS) & Mv_FDS > 0)
mv_pos_formula <- as.formula(paste("Mv_FDS ~ cluster +", paste(confounders, collapse = " + ")))

pos_model_MV <- NULL
if(nrow(df_pos_mv) == 0) {
  message("没有 Mv_FDS > 0 的样本，跳过正值部分建模。")
} else {
  if(is_integer_like(df_pos_mv$Mv_FDS)) {
    pos_model_MV <- tryCatch(glm.nb(mv_pos_formula, data = df_pos_mv),
                             error = function(e) { message("glm.nb for MV failed, fallback to lm: ", e$message); lm(mv_pos_formula, data = df_pos_mv) })
  } else {
    pos_model_MV <- tryCatch(glm(mv_pos_formula, data = df_pos_mv, family = Gamma(link = "log")),
                             error = function(e) { message("Gamma glm for MV failed, fallback to log-lm: ", e$message);
                               lm(as.formula(paste("log(Mv_FDS) ~ cluster +", paste(confounders, collapse = " + "))), data = df_pos_mv) })
  }
  capture.output({
    cat("==== Mv_FDS positive-part model summary ====\n")
    print(summary(pos_model_MV))
  }, file = file.path(outdir, "MV_pos_model_summary.txt"))
  
  safe_tidy_mv <- tryCatch(broom::tidy(pos_model_MV), error = function(e) NULL)
  if(!is.null(safe_tidy_mv)) write.csv(safe_tidy_mv, file.path(outdir, "MV_pos_model_tidy.csv"), row.names = FALSE)
  
  # 若使用 log-link，保存 exponentiated effects (近似)
  fam_mv <- if(inherits(pos_model_MV, "glm")) pos_model_MV$family else NULL
  if(!is.null(fam_mv) && tolower(fam_mv$family) == "gamma" && tolower(fam_mv$link) == "log") {
    coef_tab_mv <- broom::tidy(pos_model_MV) %>%
      mutate(
        exp_est = exp(estimate),
        exp_low = exp(estimate - 1.96 * std.error),
        exp_high = exp(estimate + 1.96 * std.error)
      )
    write.csv(coef_tab_mv, file.path(outdir, "MV_pos_model_exponentiated_effects.csv"), row.names = FALSE)
  }
}

# ---- 7. Fine-Gray 竞争风险分析（首次出 ICU 为事件，死亡为竞争事件） ----
# 构造 time_fg / status_fg（请确认此逻辑与你数据语义一致）
df <- df %>%
  mutate(
    time_fg = ifelse(day_28_mortality == 1, pmin(Survival_Time, 28, na.rm = TRUE),
                     ifelse(!is.na(ICU_LOS) & ICU_LOS <= 28, ICU_LOS, 28)),
    status_fg = ifelse(day_28_mortality == 1, 2,
                       ifelse(!is.na(ICU_LOS) & ICU_LOS <= 28, 1, 0))
  )

table(df$status_fg, useNA = "ifany")

# 准备协变量矩阵（注意去掉第一列截距）
fg_formula <- as.formula(paste("~ cluster +", paste(confounders, collapse = " + ")))
covmat <- tryCatch(model.matrix(fg_formula, data = df)[, -1, drop = FALSE], error = function(e) NULL)

if(is.null(covmat) || ncol(covmat) == 0) {
  warning("无法构建 Fine-Gray 的 covariate 矩阵（可能所有协变量缺失或无效）。跳过 Fine-Gray 分析。")
} else {
  ftime <- df$time_fg
  fstatus <- df$status_fg
  fg_model <- tryCatch(
    crr(ftime = ftime, fstatus = fstatus, cov1 = covmat, failcode = 1, cencode = 0),
    error = function(e) { message("Fine-Gray 拟合失败: ", e$message); NULL }
  )
  capture.output({
    cat("==== Fine-Gray summary ====\n")
    if(is.null(fg_model)) cat("Fine-Gray 模型拟合失败或被跳过\n") else print(summary(fg_model))
  }, file = file.path(outdir, "FineGray_ICUdischarge_summary.txt"))
  
  if(!is.null(fg_model)) {
    fg_coefs <- data.frame(coef = fg_model$coef,
                           se = sqrt(diag(fg_model$var)),
                           z = fg_model$coef / sqrt(diag(fg_model$var)),
                           p = 2 * pnorm(-abs(fg_model$coef / sqrt(diag(fg_model$var)))))
    write.csv(fg_coefs, file.path(outdir, "FineGray_coeffs.csv"), row.names = TRUE)
  }
}

# ---- 8. 导出其它 tidies（如有） ----
# ICU_zero / MV_zero 已在上面导出；positive 部分的 tidy 也已导出（如成功）
# pos_model_ICU, pos_model_MV 若存在，其 tidy 文件名已写出

message("脚本执行完毕。请检查输出目录：", outdir)
message("注意：hurdle/zeroinfl 已被注释/省略（不适合连续 FDS）。如果你确实要运行这些模型，请先确保 FDS 为整数计数并启用 pscl 包。")


# 若你希望，我可以：
# - 帮你把 hurdle/zeroinfl 的结果写成可直接放进论文的 Results 段落；或
# - 帮你把模型诊断（残差 / AIC /拟合优度）与图形输出丰富化。
# ---- 补充：logistic (zero part) 的 OR 导出 ----
if(!is.null(glm_zero)) {
  or_tab <- broom::tidy(glm_zero) %>%
    mutate(
      OR = exp(estimate),
      OR_low = exp(estimate - 1.96 * std.error),
      OR_high = exp(estimate + 1.96 * std.error)
    )
  write.csv(or_tab,
            file.path(outdir, "ICU_zero_logistic_exponentiated_effects.csv"),
            row.names = FALSE)
}

if(!is.null(glm_zero_mv)) {
  or_tab_mv <- broom::tidy(glm_zero_mv) %>%
    mutate(
      OR = exp(estimate),
      OR_low = exp(estimate - 1.96 * std.error),
      OR_high = exp(estimate + 1.96 * std.error)
    )
  write.csv(or_tab_mv,
            file.path(outdir, "MV_zero_logistic_exponentiated_effects.csv"),
            row.names = FALSE)
}
