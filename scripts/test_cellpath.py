# In[1]
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


# In[2] Test on Cycle-tree

num_metacells = 400
num_trajs = 4
kt_dpt = []
kt_cellpath = []
kt_slingshot = []

path = "../data/sim/Symsim/"

for i in range(1,11):
# for i in [1,2,3,4,9,10]:
    adata = anndata.read_h5ad(path + "cycletree_rand" + str(i) + ".h5ad")
    print(adata.shape)
    pt_true = adata.obs["sim_time"].values

    cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
    # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
    # cellpath_obj.all_in_one(flavor = "hier", num_metacells = num_metacells, resolution = None, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "fast")
    # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = None, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "exact")
    cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
    cellpath_obj.meta_cell_construction(flavor = "k-means", n_clusters = num_metacells)
    cellpath_obj.meta_cell_graph(k_neighs = 13, pruning = True, distance_scalar = 1)
    cellpath_obj.meta_paths_finding(threshold = 0.5, cutoff_length = 5, length_bias = 0.7, mode = "fast")
    cellpath_obj.first_order_pt(num_trajs = num_trajs, prop_insert = 0.700)
    visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(20,20), save_as= None)
    kt_cellpath.extend([x for x in bmk.cellpath_kt(cellpath_obj).values()])

np.save("../results/cellpath/kt_cellpath_cycletree_fast.npy", kt_cellpath)

# In[3] Test on Multicycles

num_metacells = 80
num_trajs = 1
kt_dpt = []
kt_cellpath = []
kt_slingshot = []

path = "../data/sim/Symsim/"

for i in range(1,6):
    adata = anndata.read_h5ad(path + "multi_cycles_200_rand" + str(i) + ".h5ad")
    print(adata.shape)
    pt_true = adata.obs["sim_time"].values

    cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
    # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
    # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = None, n_neighs = 5, num_trajs = num_trajs, insertion = True, prop_insert = 0.30, seed = 0, mode = "exact", pruning = False, scaling = 4, distance_scalar = 0.5)
    cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = None, n_neighs = 5, num_trajs = num_trajs, insertion = True, prop_insert = 0.70, seed = 0, mode = "fast", pruning = False, scaling = 4, distance_scalar = 0.5)

    visual.first_order_approx_pt(cellpath_obj, basis="pca", trajs = num_trajs, figsize=(10,5), save_as= None)

    kt_cellpath.extend([x for x in bmk.cellpath_kt(cellpath_obj).values()])

np.save("../results/cellpath/kt_cellpath_multicycles_80_fast.npy", kt_cellpath)


# In[5] Test on Dyngen Tree, velocity stochastic model.
path = "../data/sim/dyngen_tree/"
use_dynamical = False
num_trajs = 3
num_metacells = 200
kt_cellpath = []
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
    
    pt_true = adata.obs["sim_time"].values

    cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
    # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
    # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.3, seed = 0, mode = "exact", pruning = True, scaling = 4, distance_scalar = 0.5)
    cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

    visual.first_order_approx_pt(cellpath_obj, basis="umap", trajs = num_trajs, figsize=(10,20), save_as= None)
    kt_cellpath.extend([x for x in bmk.cellpath_kt(cellpath_obj).values()])

np.save("../results/cellpath/kt_cellpath_dyngen_tree_fast.npy", kt_cellpath)
  

# In[6] Test on VeloSim tree, using precalculated (stochastic better result)
path = "../data/sim/velosim_tree/"
# construct anndata
num_metacells = 200
kt_cellpath = []
num_trajs = 3

datasets = ["3branches_rand0", "3branches_rand1", "3branches_rand3", "3branches_rand4", "4branches_rand0"]
# datasets = ["3branches_rand3"]
for dataset in datasets:
    """
    counts_u = pd.read_csv(path + dataset + "/counts_u.csv").values.T
    counts_s = pd.read_csv(path + dataset + "/counts_s.csv").values.T
    velocity = pd.read_csv(path + dataset + "/velocity.csv").values.T
    pt = pd.read_csv(path + dataset + "/pseudo_time.csv")
    bb = pd.read_csv(path + dataset + "/backbone.csv")
    adata = anndata.AnnData(X = counts_s)
    
    adata.obs.index = ["Cell_" + str(x) for x in pt.index]
    adata.obs["sim_time"] = pt.values
    adata.obs["pop"] = bb.values
    adata.layers["unspliced"] = counts_u
    adata.layers["spliced"] = counts_s
    adata.layers["true_velocity"] = velocity
    scv.pp.filter_and_normalize(adata)
    scv.tl.velocity(adata, mode = "stochastic")
    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding(adata, basis = "pca", color = "pop", arrow_length = 5, figsize = (20,10))
    adata.write_h5ad(path + dataset + ".h5ad")
    """
    
    adata = anndata.read_h5ad(path + dataset + ".h5ad")
    X_pca = adata.obsm["X_pca"]
    pt_true = adata.obs["sim_time"].values
    cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
    # here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
    # cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.3, seed = 0, mode = "exact", pruning = True, scaling = 4, distance_scalar = 0.5)
    cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)
    cellpath_obj.adata.obsm["X_pca"] = X_pca

    visual.first_order_approx_pt(cellpath_obj, basis="pca", trajs = num_trajs, figsize=(20,10), save_as= None)
    kt_cellpath.extend([x for x in bmk.cellpath_kt(cellpath_obj).values()])

np.save("../results/cellpath/kt_cellpath_velosim_tree_fast.npy", np.array(kt_cellpath))









# %%
