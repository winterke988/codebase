rm(list = ls())
gc()
setwd("E:\\100-科研\\667_总数据版本\\004NBclust")

getwd()
library(readr)


library(NbClust)
library(data.table)

# ----- development ----- #
# load data
input_data <- read.csv("E:\\100-科研\\667_总数据版本\\003对数转换\\scaled_for_clustering_with_ID.csv")


# 假设 'ssid' 是病人 ID —— 保留下来
id_col <- input_data$id

# 去掉 ID，只对其他变量做聚类
data_mtx <- data.matrix(subset(input_data, select = -id))
#data_mtx <- data.matrix(subset(input_data))
# 运行 NbClust
res <- NbClust(data_mtx, diss = NULL, distance = 'euclidean', min.nc =2, max.nc = 8, 
               method = 'ward.D2', index = 'all')

# 聚类标签（res$Best.partition）是一个向量，顺序对应 data_mtx 的行
cluster_labels <- res$Best.partition

# 合并回病人 ID
output_df <- data.frame(ssid = id_col, cluster = cluster_labels)




write.csv(as.data.frame(t(as.matrix(res$All.index))), 
          'E:\\100-科研\\666_必胜版\\002Nbclust\\nbclust_index.csv')

write.csv(res$Best.partition, 
          'E:\\100-科研\\666_必胜版\\002Nbclust\\nbclust_label.csv')

apply(data_mtx, 2, var)
