# 清空环境
rm(list = ls())
gc()
setwd("E:\\100-科研\\667_总数据版本\\008用于做和弦图的数据\\和弦图2_0815")
getwd()
library(readr)
library(dplyr)
library(circlize)

# 读取数据
lab_df <- read.csv("E:\\100-科研\\667_总数据版本\\008用于做和弦图的数据\\chord_lab_median_LPA_cluster.csv")

# 重命名聚类
lab_df$from <- recode(lab_df$from, 
                      "Cluster 1" = "I",
                      "Cluster 2" = "II",
                      "Cluster 3" = "III",
                      "Cluster 4" = "IV")

# 定义顺序
orders <- c('I', 'II', 'III', "IV",
            'Inflammation', 'Coagulation', 'Hematologic',
            'Circulation', 'Metabolic', 'Hepatic', 'Renal', 'Respiratory')

# 定义颜色
grid.col <- c(
  'IV' = '#CE4257',
  'I' = '#F9C77E',
  'II' = '#7B967A',
  'III' = '#79A3D9',
  'Inflammation' = '#DCDCDC',
  'Hepatic' = '#DCDCDC',
  'Hematologic' = '#DCDCDC',
  'Circulation' = '#DCDCDC',
  'Metabolic' = '#DCDCDC',
  'Respiratory' = '#DCDCDC',
  'Renal' = '#DCDCDC',
  'Coagulation' = '#DCDCDC'
)

# 设置字体
par(family = "sans")  # 使用系统默认的无衬线字体（通常Arial是Windows默认）


# 创建 SVG
svg("chord_diagram_LPA.svg", 
    width = 10, 
    height = 10,
    family = "Arial",
    onefile = TRUE,
    bg = "transparent",
    antialias = "default")

# orders 有 12 个元素，所以 gap.after 也要 12 个
circos.par(
  gap.after = c(8, 8, 8, 12,   # I, II, III, IV
                6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
)

# 绘制和弦图（去掉默认 name）
chordDiagram(lab_df, 
             order = orders, 
             grid.col = grid.col, 
             scale = FALSE, 
             annotationTrack = c('grid'),
             annotationTrackHeight = c(0.05, 0.05),
             preAllocateTracks = list(track.height = 0.07))

# 标签
circos.track(track.index = 1, panel.fun = function(x, y) {
  sector.name = get.cell.meta.data("sector.index")
  circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
              facing = "bending", niceFacing = FALSE,
              adj = c(0.5, 0),
              cex = 1.5, font = 2)
}, bg.border = NA)

dev.off()



# 定义颜色映射函数
create_grid_col <- function(highlight) {
  c(
    'I' = ifelse(highlight == "I", '#F9C77E', '#DCDCDC'),
    'II' = ifelse(highlight == "II", '#7B967A', '#DCDCDC'),
    'III' = ifelse(highlight == "III", '#79A3D9', '#DCDCDC'),
    'IV' = ifelse(highlight == "IV", '#CE4257', '#DCDCDC'),
    'Inflammation' = '#DCDCDC',
    'Hepatic' = '#DCDCDC',
    'Hematologic' = '#DCDCDC',
    'Circulation' = '#DCDCDC',
    'Metabolic' = '#DCDCDC',
    'Respiratory' = '#DCDCDC',
    'Renal' = '#DCDCDC',
    'Coagulation' = '#DCDCDC'
  )
}

# 绘图函数（更新版本）
plot_chord <- function(highlight_cluster, filename) {
  library(circlize)
  
  # 清空上一次绘图
  circos.clear()
  
  # 设置全局字体
  par(family = "sans")
  
  # 创建颜色映射
  grid.col <- create_grid_col(highlight_cluster)
  
  # 创建 SVG
  svg(filename, 
      width = 10, 
      height = 10,
      family = "Arial",
      onefile = TRUE,
      bg = "transparent",
      antialias = "default")
  
  # 设置 gap.after（长度必须和 orders 一致）
  circos.par(
    gap.after = c(8, 8, 8, 12,    # I, II, III, IV
                  6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
  )
  
  # 绘制和弦图（去掉默认 name）
  chordDiagram(lab_df, 
               order = orders, 
               grid.col = grid.col, 
               scale = FALSE, 
               annotationTrack = c('grid'),
               annotationTrackHeight = c(0.05, 0.05),
               preAllocateTracks = list(track.height = 0.07))
  
  # 添加自定义标签
  circos.track(track.index = 1, panel.fun = function(x, y) {
    sector.name = get.cell.meta.data("sector.index")
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
                facing = "bending", niceFacing = FALSE,
                adj = c(0.5, 0),
                cex = 1.5, font = 2)
  }, bg.border = NA)
  
  # 关闭文件
  dev.off()
}

# 生成每个聚类的图
plot_chord("I", "chord_diagram_LPA_I.svg")
plot_chord("II", "chord_diagram_LPA_II.svg")
plot_chord("III", "chord_diagram_LPA_III.svg")
plot_chord("IV", "chord_diagram_LPA_IV.svg")







































# 读取数据
lab_df <- read.csv("E:\\100-科研\\667_总数据版本\\008用于做和弦图的数据\\chord_lab_median_GMM_cluster.csv")

# 重命名聚类
lab_df$from <- recode(lab_df$from, 
                      "Cluster 1" = "I",
                      "Cluster 2" = "II",
                      "Cluster 3" = "III",
                      "Cluster 4" = "IV")

# 定义顺序
orders <- c('I', 'II', 'III', "IV",
            'Inflammation', 'Coagulation', 'Hematologic',
            'Circulation', 'Metabolic', 'Hepatic', 'Renal', 'Respiratory')

# 定义颜色
grid.col <- c(
  'IV' = '#CE4257',
  'I' = '#F9C77E',
  'II' = '#7B967A',
  'III' = '#79A3D9',
  'Inflammation' = '#DCDCDC',
  'Hepatic' = '#DCDCDC',
  'Hematologic' = '#DCDCDC',
  'Circulation' = '#DCDCDC',
  'Metabolic' = '#DCDCDC',
  'Respiratory' = '#DCDCDC',
  'Renal' = '#DCDCDC',
  'Coagulation' = '#DCDCDC'
)

# 设置字体
par(family = "sans")  # 使用系统默认的无衬线字体（通常Arial是Windows默认）


# 创建 SVG
svg("chord_diagram_GMM.svg", 
    width = 10, 
    height = 10,
    family = "Arial",
    onefile = TRUE,
    bg = "transparent",
    antialias = "default")

# orders 有 12 个元素，所以 gap.after 也要 12 个
circos.par(
  gap.after = c(8, 8, 8, 12,   # I, II, III, IV
                6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
)

# 绘制和弦图（去掉默认 name）
chordDiagram(lab_df, 
             order = orders, 
             grid.col = grid.col, 
             scale = FALSE, 
             annotationTrack = c('grid'),
             annotationTrackHeight = c(0.05, 0.05),
             preAllocateTracks = list(track.height = 0.07))

# 标签
circos.track(track.index = 1, panel.fun = function(x, y) {
  sector.name = get.cell.meta.data("sector.index")
  circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
              facing = "bending", niceFacing = FALSE,
              adj = c(0.5, 0),
              cex = 1.5, font = 2)
}, bg.border = NA)

dev.off()



# 定义颜色映射函数
create_grid_col <- function(highlight) {
  c(
    'I' = ifelse(highlight == "I", '#F9C77E', '#DCDCDC'),
    'II' = ifelse(highlight == "II", '#7B967A', '#DCDCDC'),
    'III' = ifelse(highlight == "III", '#79A3D9', '#DCDCDC'),
    'IV' = ifelse(highlight == "IV", '#CE4257', '#DCDCDC'),
    'Inflammation' = '#DCDCDC',
    'Hepatic' = '#DCDCDC',
    'Hematologic' = '#DCDCDC',
    'Circulation' = '#DCDCDC',
    'Metabolic' = '#DCDCDC',
    'Respiratory' = '#DCDCDC',
    'Renal' = '#DCDCDC',
    'Coagulation' = '#DCDCDC'
  )
}

# 绘图函数（更新版本）
plot_chord <- function(highlight_cluster, filename) {
  library(circlize)
  
  # 清空上一次绘图
  circos.clear()
  
  # 设置全局字体
  par(family = "sans")
  
  # 创建颜色映射
  grid.col <- create_grid_col(highlight_cluster)
  
  # 创建 SVG
  svg(filename, 
      width = 10, 
      height = 10,
      family = "Arial",
      onefile = TRUE,
      bg = "transparent",
      antialias = "default")
  
  # 设置 gap.after（长度必须和 orders 一致）
  circos.par(
    gap.after = c(8, 8, 8, 12,    # I, II, III, IV
                  6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
  )
  
  # 绘制和弦图（去掉默认 name）
  chordDiagram(lab_df, 
               order = orders, 
               grid.col = grid.col, 
               scale = FALSE, 
               annotationTrack = c('grid'),
               annotationTrackHeight = c(0.05, 0.05),
               preAllocateTracks = list(track.height = 0.07))
  
  # 添加自定义标签
  circos.track(track.index = 1, panel.fun = function(x, y) {
    sector.name = get.cell.meta.data("sector.index")
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
                facing = "bending", niceFacing = FALSE,
                adj = c(0.5, 0),
                cex = 1.5, font = 2)
  }, bg.border = NA)
  
  # 关闭文件
  dev.off()
}

# 生成每个聚类的图
plot_chord("I", "chord_diagram_GMM_I.svg")
plot_chord("II", "chord_diagram_GMM_II.svg")
plot_chord("III", "chord_diagram_GMM_III.svg")
plot_chord("IV", "chord_diagram_GMM_IV.svg")











# 读取数据
lab_df <- read.csv("E:\\100-科研\\667_总数据版本\\008用于做和弦图的数据\\chord_lab_median_ALG_cluster.csv")

# 重命名聚类
lab_df$from <- recode(lab_df$from, 
                      "Cluster 1" = "I",
                      "Cluster 2" = "II",
                      "Cluster 3" = "III",
                      "Cluster 4" = "IV")

# 定义顺序
orders <- c('I', 'II', 'III', "IV",
            'Inflammation', 'Coagulation', 'Hematologic',
            'Circulation', 'Metabolic', 'Hepatic', 'Renal', 'Respiratory')

# 定义颜色
grid.col <- c(
  'IV' = '#CE4257',
  'I' = '#F9C77E',
  'II' = '#7B967A',
  'III' = '#79A3D9',
  'Inflammation' = '#DCDCDC',
  'Hepatic' = '#DCDCDC',
  'Hematologic' = '#DCDCDC',
  'Circulation' = '#DCDCDC',
  'Metabolic' = '#DCDCDC',
  'Respiratory' = '#DCDCDC',
  'Renal' = '#DCDCDC',
  'Coagulation' = '#DCDCDC'
)

# 设置字体
par(family = "sans")  # 使用系统默认的无衬线字体（通常Arial是Windows默认）


# 创建 SVG
svg("chord_diagram_ALG.svg", 
    width = 10, 
    height = 10,
    family = "Arial",
    onefile = TRUE,
    bg = "transparent",
    antialias = "default")

# orders 有 12 个元素，所以 gap.after 也要 12 个
circos.par(
  gap.after = c(8, 8, 8, 12,   # I, II, III, IV
                6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
)

# 绘制和弦图（去掉默认 name）
chordDiagram(lab_df, 
             order = orders, 
             grid.col = grid.col, 
             scale = FALSE, 
             annotationTrack = c('grid'),
             annotationTrackHeight = c(0.05, 0.05),
             preAllocateTracks = list(track.height = 0.07))

# 标签
circos.track(track.index = 1, panel.fun = function(x, y) {
  sector.name = get.cell.meta.data("sector.index")
  circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
              facing = "bending", niceFacing = FALSE,
              adj = c(0.5, 0),
              cex = 1.5, font = 2)
}, bg.border = NA)

dev.off()



# 定义颜色映射函数
create_grid_col <- function(highlight) {
  c(
    'I' = ifelse(highlight == "I", '#F9C77E', '#DCDCDC'),
    'II' = ifelse(highlight == "II", '#7B967A', '#DCDCDC'),
    'III' = ifelse(highlight == "III", '#79A3D9', '#DCDCDC'),
    'IV' = ifelse(highlight == "IV", '#CE4257', '#DCDCDC'),
    'Inflammation' = '#DCDCDC',
    'Hepatic' = '#DCDCDC',
    'Hematologic' = '#DCDCDC',
    'Circulation' = '#DCDCDC',
    'Metabolic' = '#DCDCDC',
    'Respiratory' = '#DCDCDC',
    'Renal' = '#DCDCDC',
    'Coagulation' = '#DCDCDC'
  )
}

# 绘图函数（更新版本）
plot_chord <- function(highlight_cluster, filename) {
  library(circlize)
  
  # 清空上一次绘图
  circos.clear()
  
  # 设置全局字体
  par(family = "sans")
  
  # 创建颜色映射
  grid.col <- create_grid_col(highlight_cluster)
  
  # 创建 SVG
  svg(filename, 
      width = 10, 
      height = 10,
      family = "Arial",
      onefile = TRUE,
      bg = "transparent",
      antialias = "default")
  
  # 设置 gap.after（长度必须和 orders 一致）
  circos.par(
    gap.after = c(8, 8, 8, 12,    # I, II, III, IV
                  6, 6, 6, 6, 6, 6, 6, 12) # 7 个器官 + 最后空隙
  )
  
  # 绘制和弦图（去掉默认 name）
  chordDiagram(lab_df, 
               order = orders, 
               grid.col = grid.col, 
               scale = FALSE, 
               annotationTrack = c('grid'),
               annotationTrackHeight = c(0.05, 0.05),
               preAllocateTracks = list(track.height = 0.07))
  
  # 添加自定义标签
  circos.track(track.index = 1, panel.fun = function(x, y) {
    sector.name = get.cell.meta.data("sector.index")
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], sector.name, 
                facing = "bending", niceFacing = FALSE,
                adj = c(0.5, 0),
                cex = 1.5, font = 2)
  }, bg.border = NA)
  
  # 关闭文件
  dev.off()
}

# 生成每个聚类的图
plot_chord("I", "chord_diagram_ALG_I.svg")
plot_chord("II", "chord_diagram_ALG_II.svg")
plot_chord("III", "chord_diagram_ALG_III.svg")
plot_chord("IV", "chord_diagram_ALG_IV.svg")





