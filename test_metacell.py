# In[0]
from scipy.sparse import data
import scvelo as scv
import numpy as np  
import matplotlib.pyplot as plt

import anndata

import sys
sys.path.append("../CellPath/")


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

# In[1]
path = "./additional_data/dyngen_tree/"
use_dynamical = True
num_metacells = 200

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

            # visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(10,20), save_as= None)
            kt_cellpath.extend([{"n_meta": num_metacells, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

            # global pt
            global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
            global_kt_cellpath.append({"n_meta": num_metacells, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)

# In[2]
import seaborn as sns
import matplotlib.pyplot as plt

# fig = plt.figure(figsize = (10,7))
# ax = fig.add_subplot()
# sns.boxplot(data = kt_cellpath, x = "n_meta", y = "kt", ax = ax)
# ax.set_title("number of meta-cell", fontsize = 20)
# fig.savefig("n_meta.png")

# kt_cellpath.to_csv("n_meta_dyngen_tree.csv")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "n_meta", y = "kt", ax = ax)
ax.set_title("number of meta-cell", fontsize = 20)
fig.savefig("n_meta_global.png")

global_kt_cellpath.to_csv("n_meta_dyngen_tree_globale.csv")




# In[3]
path = "./additional_data/dyngen_tree/"
use_dynamical = True
num_metacells = 200

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

    for distance_scalar in [0, 0.5, 1, 2]:
        num_trajs = 6
        for seed in range(10):
            adata2 = adata.copy()
            cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
            # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
            cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "exact", pruning = True, scaling = 4, distance_scalar = distance_scalar, cutoff_length = None)
            # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.30, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

            # visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(10,20), save_as= None)
            kt_cellpath.extend([{"distance_scalar": distance_scalar, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

            # global pt
            global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
            global_kt_cellpath.append({"distance_scalar": distance_scalar, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)



# In[4]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "distance_scalar", y = "kt", ax = ax)
ax.set_title("distance scalar", fontsize = 20)
fig.savefig("distance_scalar.png")

kt_cellpath.to_csv("distance_scalar_dyngen_tree.csv")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "distance_scalar", y = "kt", ax = ax)
ax.set_title("distance scalar", fontsize = 20)
fig.savefig("distance_scalar_global.png")

global_kt_cellpath.to_csv("distance_scalar_dyngen_tree_globale.csv")




# In[5]
path = "./additional_data/dyngen_tree/"
use_dynamical = True
num_metacells = 200
distance_scalar = 0.5

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

    for scaling in [1, 2, 3, 4]:
        num_trajs = 6
        for seed in range(10):
            adata2 = adata.copy()
            cellpath_obj = cp.CellPath(adata = adata2, preprocess = True)
            # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
            cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 1.0, seed = seed, mode = "exact", pruning = True, scaling = scaling, distance_scalar = distance_scalar, cutoff_length = None)
            # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.30, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

            # visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(10,20), save_as= None)
            kt_cellpath.extend([{"scaling": scaling, "dataset": rand, "kt": x} for x in bmk.cellpath_kt(cellpath_obj).values()])

            # global pt
            global_pt = np.nanmean(cellpath_obj.pseudo_order.values, axis = 1)
            global_kt_cellpath.append({"scaling": scaling, "dataset": rand, "kt": bmk.kendalltau(pt_pred = global_pt, pt_true = adata.obs["sim_time"])})


kt_cellpath = pd.DataFrame(kt_cellpath)
global_kt_cellpath = pd.DataFrame(global_kt_cellpath)

# In[6]
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = kt_cellpath, x = "scaling", y = "kt", ax = ax)
ax.set_title("scaling", fontsize = 20)
fig.savefig("scaling.png")

kt_cellpath.to_csv("scaling_dyngen_tree.csv")

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = global_kt_cellpath, x = "scaling", y = "kt", ax = ax)
ax.set_title("scaling", fontsize = 20)
fig.savefig("scaling_global.png")

global_kt_cellpath.to_csv("scaling_dyngen_tree_globale.csv")




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
