rm(list = ls())
gc()
library(anndata)
library(SingleCellExperiment)
library(mclust, quietly = TRUE)
library(RColorBrewer)
library(slingshot)
library(umap)

setwd("/Users/ziqizhang/Dropbox (GaTech)/Research/Projects/CellPath/additional_tests/")

# velosim tree
path <- "./additional_data/velosim_tree/"
datasets <- c("3branches_rand0", "3branches_rand1", "4branches_rand0", "3branches_rand3", "3branches_rand4")
rand <- 5
ad <- read_h5ad(paste0(path, datasets[rand], ".h5ad"))

# trifurcating
path <- "../CellPath/sim_data/Dyngen/"
datasets <- c("Bifurcating")
rand <- 1
ad <- read_h5ad(paste0(path, datasets[rand], ".h5ad"))

X_pca <- ad$obsm$X_pca
X_umap <- ad$obsm$X_umap
# gene by cell
counts <- t(ad$layers["spliced"])

sce <- SingleCellExperiment(assays = List(counts = counts))

reducedDims(sce) <- SimpleList(PCA = X_pca)

# GMM
# cl1 <- Mclust(X_pca)$classification
# colData(sce)$GMM <- cl1
# kmeans
num_clust <- 7
cl2 <- kmeans(X_pca, centers = num_clust)$cluster
colData(sce)$kmeans <- cl2
group <- brewer.pal(num_clust,"Paired")[cl2]
plot(X_pca, col = group, pch=16, asp = 1)
legend("topleft", legend=seq(1,num_clust), pch=16, col=brewer.pal(num_clust,"Paired")[seq(1,num_clust)])

mean_pt <- lapply(seq(num_clust), function(X){
  mean(ad$obs$sim_time[cl2 == X])
})

start <- which.min(mean_pt)

sce <- slingshot(sce, clusterLabels = 'kmeans', reducedDim = 'PCA', start.clus = start)
ss <- colData(sce)$slingshot
pt <- ss@assays@data@listData$pseudotime

# write.table(pt, file = paste0("slingshot/simulated/cycle_tree/cycletree_rand", rand, "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
# write.table(pt, file = paste0("slingshot/simulated/dyngen_tree/dyngen_tree_rand", rand, "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
# write.table(pt, file = paste0("slingshot/simulated/velosim_tree/", datasets[rand], "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(pt, file = paste0("slingshot/simulated/trifur/", datasets[rand], "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)


# path <- "../CellPath/sim_data/Symsim/"
# rand <- 5
# ad <- read_h5ad(paste0(path, "multi_cycles_200_rand", rand, ".h5ad"))
# X_pca <- ad$obsm$X_pca
# X_umap <- ad$obsm$X_umap
# # gene by cell
# counts <- t(ad$layers["spliced"])
#
# sce <- SingleCellExperiment(assays = List(counts = counts))
#
# reducedDims(sce) <- SimpleList(PCA = X_pca, UMAP = X_umap)
#
# # GMM
# # cl1 <- Mclust(X_pca)$classification
# # colData(sce)$GMM <- cl1
# # kmeans
# num_clust <- 8
# cl2 <- kmeans(X_pca, centers = num_clust)$cluster
# colData(sce)$kmeans <- cl2
# group <- brewer.pal(num_clust,"Paired")[cl2]
# plot(X_pca, col = group, pch=16, asp = 1)
# legend("topleft", legend=seq(1,num_clust), pch=16, col=brewer.pal(num_clust,"Paired")[seq(1,num_clust)])
#
# mean_pt <- lapply(seq(num_clust), function(X){
#   mean(ad$obs$sim_time[cl2 == X])
# })
#
# start <- which.min(mean_pt)
#
# sce <- slingshot(sce, clusterLabels = 'kmeans', reducedDim = 'PCA', start.clus = start)
# ss <- colData(sce)$slingshot
# pt <- ss@assays@data@listData$pseudotime
#
# # write.table(pt, file = paste0("slingshot/simulated/cycle_tree/cycletree_rand", rand, "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
# write.table(pt, file = paste0("slingshot/simulated/multi_cycle/multi_cycle_rand", rand, "_slingshot.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)



