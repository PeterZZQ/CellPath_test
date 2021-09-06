# In[0]
from scipy.sparse import data
import scvelo as scv
import numpy as np  
import matplotlib.pyplot as plt

import anndata

import sys
sys.path.append("../")


import cellpath as cp
import cellpath.visual as visual
import cellpath.benchmark as bmk 
import cellpath.de_analy as de


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import seaborn as sns

import pandas as pd
from matplotlib import rcParams
labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

# In[1]
def metacell(cellpath_obj, basis = "pca", figsize = (20,10), save_as = None, title = None, **kwargs):

    _kwargs = {
        "axis": True,
    }

    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()

    if _kwargs["axis"]:
        ax.tick_params(axis = "both", direction = "out", labelsize = 16)
        if basis == "pca":
            ax.set_xlabel("PC 1", fontsize = 19)
            ax.set_ylabel("PC 2", fontsize = 19)
        else:
            ax.set_xlabel(basis + " 1", fontsize = 19)
            ax.set_ylabel(basis + " 2", fontsize = 19)            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        ax.axis('off')

    if "X_" + basis not in cellpath_obj.adata.obsm:
        raise ValueError("please calculate " + basis + " first")
    else:
        X_cell = cellpath_obj.adata.obsm["X_"+basis].copy()
        groups = cellpath_obj.groups
        # note that when incorporate the uncovered cells, the group value is changed.
        X_clust = np.zeros((np.unique(groups).shape[0], X_cell.shape[1]))
        radius_clust = []
        for i, c in enumerate(np.unique(groups)):
            indices = np.where(groups == c)[0]
            X_clust[i,:] = np.mean(X_cell[indices,:], axis = 0)
            radius_clust.append(np.max(np.linalg.norm(X_cell[indices,:] - X_clust[i,:], axis = 1)))

        radius_clust = np.array(radius_clust)
        print(radius_clust)
        for i in range(X_clust.shape[0]):
            ax.scatter(X_clust[i,0], X_clust[i,1], color = 'gray', alpha = 0.5, s = 100 * radius_clust[i])

    # ax.scatter(X_clust[:,0], X_clust[:,1], color = 'gray', alpha = 0.5, s = 1)
    
    if title != None:
        plt.title(title)
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')



path = "../data/sim/dyngen_tree/"
use_dynamical = False
num_metacells = 200

kt_cellpath = []
global_kt_cellpath = []
rand = 2
adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
# adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# adata.layers["spliced"] = adata.layers["counts_spliced"]
# scv.pp.normalize_per_cell(adata)
# scv.pp.log1p(adata)
# scv.tl.umap(adata)
if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 1, color = "sim_time")
# scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

for num_metacells in [40, 200, 500]:
    if num_metacells == 40:
        num_trajs = 4
    else:
        num_trajs = 6
    for seed in range(10):
        seed = 1
        adata2 = adata.copy()
        cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
        # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
        # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "exact", pruning = True, scaling = 4, distance_scalar = 0.5, cutoff_length = None)
        cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)            
        visual.first_order_approx_pt(cellpath_obj, basis = "umap", trajs = num_trajs, figsize = (20, 14), save_as = "../results/plots/meta_" + str(num_metacells) + "_seed_" + str(seed) + "_cellpath.png")
        # metacell(cellpath_obj, basis = "umap", figsize = (10,7), save_as = "../results/plots/meta_" + str(num_metacells) + "_seed_" + str(seed) + ".png", title = None)
        
        kt_cellpath.extend([{"n_meta": num_metacells, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

        # global pt
        global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
        global_kt_cellpath.append({"n_meta": num_metacells, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})
        break
    break

kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)

# In[2]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "n_meta", y = "kt", ax = ax)
ax.set_xlabel("number of meta-cell", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
# ax.set_title("number of meta-cell", fontsize = 20)
fig.savefig("../results/plots/n_meta.png", bbox_inches = "tight")

kt_cellpath.to_csv("../results/hyper-paras/n_meta_dyngen_tree.csv")

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "n_meta", y = "kt", ax = ax)
ax.set_xlabel("number of meta-cell", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
# ax.set_title("number of meta-cell", fontsize = 20)
fig.savefig("../results/plots/n_meta_global.png", bbox_inches = "tight")

global_kt_cellpath.to_csv("../results/hyper-paras/n_meta_dyngen_tree_globale.csv")




# In[3]
path = "../data/sim/dyngen_tree/"
use_dynamical = False
num_metacells = 200

kt_cellpath = []
global_kt_cellpath = []
rand = 2  
adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
# adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# adata.layers["spliced"] = adata.layers["counts_spliced"]
# scv.pp.normalize_per_cell(adata)
# scv.pp.log1p(adata)
# scv.tl.umap(adata)
if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 1, color = "sim_time")
# scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

for distance_scalar in [0, 0.5, 1, 2]:
    num_trajs = 6
    for seed in range(10):
        adata2 = adata.copy()
        cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
        # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
        cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "fast", pruning = True, scaling = 4, distance_scalar = distance_scalar, cutoff_length = None)
        # if seed == 0:
        #     visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(20,20), save_as= "../results/plots/dist_scalar_" + str(distance_scalar) + "_seed_" + str(seed) + "_cellpath.png")
        kt_cellpath.extend([{"distance_scalar": distance_scalar, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

        # global pt
        global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
        global_kt_cellpath.append({"distance_scalar": distance_scalar, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)



# In[4]
import seaborn as sns
import matplotlib.pyplot as plt

# kt_cellpath = pd.read_csv("../results/hyper-paras/distance_scalar_dyngen_tree.csv")

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "distance_scalar", y = "kt", ax = ax)
ax.set_xlabel("beta", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig("../results/plots/distance_scalar.png", bbox_inches = "tight")

kt_cellpath.to_csv("../results/hyper-paras/distance_scalar_dyngen_tree.csv")

# global_kt_cellpath = pd.read_csv("../results/hyper-paras/distance_scalar_dyngen_tree_globale.csv")

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "distance_scalar", y = "kt", ax = ax)
ax.set_xlabel("beta", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig("../results/plots/distance_scalar_global.png", bbox_inches = "tight")

global_kt_cellpath.to_csv("../results/hyper-paras/distance_scalar_dyngen_tree_globale.csv")




# In[5]
path = "../data/sim/dyngen_tree/"
use_dynamical = False
num_metacells = 200
distance_scalar = 0.5

kt_cellpath = []
global_kt_cellpath = []
rand = 2  
adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
# adata.layers["unspliced"] = adata.layers["counts_unspliced"]
# adata.layers["spliced"] = adata.layers["counts_spliced"]
# scv.pp.normalize_per_cell(adata)
# scv.pp.log1p(adata)
# scv.tl.umap(adata)
if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 1, color = "sim_time")
# scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

for scaling in [1, 2, 3, 4]:
    num_trajs = 6
    for seed in range(10):
        adata2 = adata.copy()
        cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
        # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
        cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "fast", pruning = True, scaling = scaling, distance_scalar = distance_scalar, cutoff_length = None)
        # if seed == 0:
        #     visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(20,20), save_as= "../results/plots/scaling_" + str(scaling) + "_seed_" + str(seed) + "_cellpath.png")
        kt_cellpath.extend([{"scaling": scaling, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

        # global pt
        global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
        global_kt_cellpath.append({"scaling": scaling, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)

# In[6]
import seaborn as sns
import matplotlib.pyplot as plt

# kt_cellpath = pd.read_csv("../results/hyper-paras/scaling_dyngen_tree.csv")
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "scaling", y = "kt", ax = ax)
ax.set_xlabel("gamma", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig("../results/plots/scaling.png", bbox_inches = "tight")

kt_cellpath.to_csv("../results/hyper-paras/scaling_dyngen_tree.csv")


# global_kt_cellpath = pd.read_csv("../results/hyper-paras/scaling_dyngen_tree_globale.csv")
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "scaling", y = "kt", ax = ax)
ax.set_xlabel("gamma", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig("../results/plots/scaling_global.png", bbox_inches = "tight")

global_kt_cellpath.to_csv("../results/hyper-paras/scaling_dyngen_tree_globale.csv")




# In[7]
path = "./additional_data/dyngen_tree/"
use_dynamical = True
num_metacells = 200
distance_scalar = 0.5
scaling = 4

kt_cellpath = []
global_kt_cellpath = []
for rand in range(1,5):   
    adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
    # adata.layers["unspliced"] = adata.layers["counts_unspliced"]
    # adata.layers["spliced"] = adata.layers["counts_spliced"]
    # scv.pp.normalize_per_cell(adata)
    # scv.pp.log1p(adata)
    # scv.tl.umap(adata)
    if use_dynamical:
        scv.tl.recover_dynamics(adata, n_jobs = 12)
        scv.tl.velocity(adata, mode = "dynamical")
        gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
        adata = adata[:,gene_idx]
    else:
        scv.tl.velocity(adata, mode = "stochastic")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 1, color = "sim_time")
    # scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

    for method in ["k-means", "hier"]:
        num_trajs = 6
        for seed in range(5):
            adata2 = adata.copy()
            cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
            # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
            cellpath_obj.all_in_one(flavor = method, num_metacells = num_metacells, resolution = 20, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "exact", pruning = True, scaling = scaling, distance_scalar = distance_scalar, cutoff_length = None)
            # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.30, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

            # visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(10,20), save_as= None)
            kt_cellpath.extend([{"method": method, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

            # global pt
            global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
            global_kt_cellpath.append({"method": method, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)

# In[8]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "method", y = "kt", ax = ax)
ax.set_title("method", fontsize = 20)
fig.savefig("method.png")

kt_cellpath.to_csv("method_dyngen_tree.csv")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "method", y = "kt", ax = ax)
ax.set_title("method", fontsize = 20)
fig.savefig("method_global.png")

global_kt_cellpath.to_csv("method_dyngen_tree_globale.csv")













# In[4]
path = "./additional_data/velosim_tree/"
use_dynamical = False
num_metacells = 200

kt_cellpath = []
datasets = ["3branches_rand0", "3branches_rand1", "4branches_rand0"]
for dataset in datasets:   
    adata = anndata.read_h5ad(path + dataset + ".h5ad")
    # adata.layers["unspliced"] = adata.layers["counts_unspliced"]
    # adata.layers["spliced"] = adata.layers["counts_spliced"]
    # scv.pp.normalize_per_cell(adata)
    # scv.pp.log1p(adata)
    # scv.tl.umap(adata)
    if use_dynamical:
        scv.tl.recover_dynamics(adata, n_jobs = 12)
        scv.tl.velocity(adata, mode = "dynamical")
        gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
        adata = adata[:,gene_idx]
    
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 1, color = "sim_time")
    # scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

    for num_metacells in [40, 200, 500]:
        if num_metacells == 40:
            num_trajs = 4
        else:
            num_trajs = 6
        for seed in range(10):
            adata2 = adata.copy()
            cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
            # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
            cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "exact", pruning = True, scaling = 4, distance_scalar = 0.5, cutoff_length = None)
            # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.30, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

            # visual.first_order_approx_pt(cellpath_obj, basis="pca", trajs = num_trajs, figsize=(10,10), save_as= None)
            kt_cellpath.extend([{"n_meta": num_metacells, "dataset": dataset, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

kt_cellpath = pd.DataFrame(kt_cellpath)


# In[4]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "n_meta", y = "kt", ax = ax)
ax.set_title("number of meta-cell", fontsize = 20)
fig.savefig("n_meta_velosim_tree.png")

kt_cellpath.to_csv("n_meta_velosim_tree.csv")


# %%
