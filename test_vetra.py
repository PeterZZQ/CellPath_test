# scvelo calculate the nonlinear velocity embedding
# In[0]
import numpy as np
import scvelo as scv
import anndata
import os
import VeTra as vt



# In[1] Multi-cycles-1
path = "../CellPath/sim_data/Symsim/"

path2 = "./vetra/simulated/multi_cycle/"
if not os.path.exists(path2):
    os.makedirs(path2)

for rand in range(1,6):
    adata = anndata.read_h5ad(path + "multi_cycles_200_rand" + str(rand) + ".h5ad")
    scv.tl.umap(adata, min_dist = 0.8)
    scv.tl.velocity_embedding(adata, basis = "pca")
    scv.tl.velocity_embedding(adata, basis = "umap")
    scv.pl.velocity_embedding(adata, basis = "pca", arrow_length = 5, color = "sim_time")
    scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 5, color = "sim_time")
    X_pca = adata.obsm["X_pca"]
    X_umap = adata.obsm["X_umap"]
    V_pca = adata.obsm["velocity_pca"]
    V_umap = adata.obsm["velocity_umap"]
    np.savetxt(path2 + "embedding_multicycle_rand" + str(rand) + "_pca.txt", X = X_pca)
    np.savetxt(path2 + "embedding_multicycle_rand" + str(rand) + "_umap.txt", X = X_umap)
    np.savetxt(path2 + "delta_embedding_multicycle_rand" + str(rand) + "_pca.txt", X = V_pca)
    np.savetxt(path2 + "delta_embedding_multicycle_rand" + str(rand) + "_umap.txt", X = V_umap)

# In[2] Multi-cycles-2
import VeTra as vt
num_traj = 1
rand = 5
ex1 = vt.VeTra(path2 + "embedding_multicycle_rand" + str(rand) + "_umap.txt", path2 + "delta_embedding_multicycle_rand" + str(rand) + "_umap.txt")
ex1.vetra(deltaThreshold = 12, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 2)


# In[3] Cycle-tree-1
path = "../CellPath/sim_data/Symsim/"
path2 = "./vetra/simulated/cycle_tree/"
if not os.path.exists(path2):
    os.makedirs(path2)

for rand in range(1,11):
    adata = anndata.read_h5ad(path + "cycletree_rand" + str(rand) + ".h5ad")
    scv.tl.umap(adata, min_dist = 0.8)
    print(adata)

    scv.tl.velocity_embedding(adata, basis = "pca")
    scv.tl.velocity_embedding(adata, basis = "umap")
    scv.pl.velocity_embedding(adata, basis = "pca", arrow_length = 5, color = "sim_time")
    scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 5, color = "sim_time")
    X_pca = adata.obsm["X_pca"]
    X_umap = adata.obsm["X_umap"]
    V_pca = adata.obsm["velocity_pca"]
    V_umap = adata.obsm["velocity_umap"]
    np.savetxt(path2 + "embedding_cycletree_rand" + str(rand) + "_pca.txt", X = X_pca)
    np.savetxt(path2 + "embedding_cycletree_rand" + str(rand) + "_umap.txt", X = X_umap)
    np.savetxt(path2 + "delta_embedding_cycletree_rand" + str(rand) + "_pca.txt", X = V_pca)
    np.savetxt(path2 + "delta_embedding_cycletree_rand" + str(rand) + "_umap.txt", X = V_umap)

# In[4] Cycle-tree-2
num_traj = 4
rand = 10
ex1 = vt.VeTra(path2 + "embedding_cycletree_rand" + str(rand) + "_umap.txt", path2 + "delta_embedding_cycletree_rand" + str(rand) + "_umap.txt")
ex1.vetra(deltaThreshold = 12, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 2)


# In[6] Bifur

path = "../CellPath/sim_data/Dyngen/"

path2 = "./vetra/simulated/bifur/"
if not os.path.exists(path2):
    os.makedirs(path2)

adata = anndata.read_h5ad(path + "Bifurcating.h5ad")
scv.tl.umap(adata)
print(adata)

scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")
scv.pl.velocity_embedding(adata, basis = "pca", arrow_length = 5, color = "sim_time")
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 5, color = "sim_time")
X_pca = adata.obsm["X_pca"]
X_umap = adata.obsm["X_umap"]
V_pca = adata.obsm["velocity_pca"]
V_umap = adata.obsm["velocity_umap"]
np.savetxt(path2 + "embedding_bifur_pca.txt", X = X_pca)
np.savetxt(path2 + "embedding_bifur_umap.txt", X = X_umap)
np.savetxt(path2 + "delta_embedding_bifur_pca.txt", X = V_pca)
np.savetxt(path2 + "delta_embedding_bifur_umap.txt", X = V_umap)


num_traj = 4

ex1 = vt.VeTra(path2 + "embedding_bifur_umap.txt", path2 + "delta_embedding_bifur_umap.txt")
ex1.vetra(deltaThreshold = 12, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 10)

# In[7] Dyngen-tree

use_dynamical = False
path = "./additional_data/dyngen_tree/"
path2 = "./vetra/simulated/dyngen_tree/"
rand = 4
# calculate velocity
adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
# adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# adata.layers["spliced"] = adata.layers["counts_spliced"]
# scv.pp.normalize_per_cell(adata)
# scv.pp.log1p(adata)
if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")

scv.tl.umap(adata, min_dist = 0.8)
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 5, color = "sim_time")

# save the file for vetra calculation
X_pca = adata.obsm["X_pca"]
X_umap = adata.obsm["X_umap"]
V_pca = adata.obsm["velocity_pca"]
V_umap = adata.obsm["velocity_umap"]
np.savetxt(path2 + "embedding_rand" + str(rand) + "_pca.txt", X = X_pca)
np.savetxt(path2 + "embedding_rand" + str(rand) + "_umap.txt", X = X_umap)
np.savetxt(path2 + "delta_embedding_rand" + str(rand) + "_pca.txt", X = V_pca)
np.savetxt(path2 + "delta_embedding_rand" + str(rand) + "_umap.txt", X = V_umap)

num_traj = 3
ex1 = vt.VeTra(path2 + "embedding_rand" + str(rand) + "_umap.txt", path2 + "delta_embedding_rand" + str(rand) + "_umap.txt")
ex1.vetra(deltaThreshold = 15, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 2)


# In[8] velosim-tree
import numpy as np
import scvelo as scv
import anndata
import os
import VeTra as vt

use_dynamical = False
path = "./additional_data/velosim_tree/"
path2 = "./vetra/simulated/velosim_tree/"
datasets = ["3branches_rand0", "3branches_rand1", "3branches_rand3", "3branches_rand4", "4branches_rand0"]
num_traj = 4
rand = 4
# calculate velocity
adata = anndata.read_h5ad(path  + datasets[rand] + ".h5ad")
# adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# adata.layers["spliced"] = adata.layers["counts_spliced"]
# scv.pp.normalize_per_cell(adata)
# scv.pp.log1p(adata)

# assert False

if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")
scv.tl.umap(adata, min_dist = 0.5)
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")
# save the file for vetra calculation

X_umap = adata.obsm["X_umap"]
V_umap = adata.obsm["velocity_umap"]

# np.savetxt(path2 + "embedding_rand" + str(rand) + "_pca.txt", X = X_pca)
np.savetxt(path2 + "embedding_rand" + str(rand) + "_umap.txt", X = X_umap)
# np.savetxt(path2 + "delta_embedding_rand" + str(rand) + "_pca.txt", X = X_pca)
np.savetxt(path2 + "delta_embedding_rand" + str(rand) + "_umap.txt", X = V_umap)

ex1 = vt.VeTra(path2 + "embedding_rand" + str(rand) + "_umap.txt", path2 + "delta_embedding_rand" + str(rand) + "_umap.txt")
ex1.vetra(deltaThreshold = 15, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 5)







# # In[9] Dentate-gyrus
# import numpy as np
# import scvelo as scv
# import anndata
# import os
# import VeTra as vt

# use_dynamical = False
# path = "./additional_data/DentateGyrus/"
# path2 = "./vetra/real/DentateGyrus/"
# datasets = ["10X43_1"]
# num_traj = 7
# rand = 0
# # calculate velocity
# adata = anndata.read_h5ad(path  + datasets[rand] + ".h5ad")
# # adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# # adata.layers["spliced"] = adata.layers["counts_spliced"]
# # scv.pp.normalize_per_cell(adata)
# # scv.pp.log1p(adata)

# # assert False
# scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
# scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# if use_dynamical:
#     scv.tl.recover_dynamics(adata, n_jobs = 12)
#     scv.tl.velocity(adata, mode = "dynamical")
#     gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
#     adata = adata[:,gene_idx]
# else:
#     scv.tl.velocity(adata, mode = "stochastic")
# scv.tl.umap(adata, min_dist = 0.8)
# scv.tl.velocity_graph(adata)
# scv.tl.velocity_embedding(adata, basis = "pca")
# scv.tl.velocity_embedding(adata, basis = "umap")
# # save the file for vetra calculation

# X_umap = adata.obsm["X_umap"]
# V_umap = adata.obsm["velocity_umap"]

# # np.savetxt(path2 + "embedding_rand" + str(rand) + "_pca.txt", X = X_pca)
# np.savetxt(path2 + "embedding_dg_umap.txt", X = X_umap)
# # np.savetxt(path2 + "delta_embedding_rand" + str(rand) + "_pca.txt", X = X_pca)
# np.savetxt(path2 + "delta_embedding_dg_umap.txt", X = V_umap)

# ex1 = vt.VeTra(path2 + "embedding_dg_umap.txt", path2 + "delta_embedding_dg_umap.txt")
# ex1.vetra(deltaThreshold = 15, WCCsizeCutoff = 5, clusternumber = num_traj, cosine_thres = 0.7, expand = 5)



# %%


