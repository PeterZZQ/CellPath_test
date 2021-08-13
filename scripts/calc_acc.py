
# In[0]
from numpy.core.defchararray import index
import pandas as pd 
import numpy as np
import anndata

import scvelo as scv
import numpy as np  
import matplotlib.pyplot as plt

from matplotlib import rcParams

labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

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

def kendalltau(pt_pred, pt_true):
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    from scipy.stats import kendalltau
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau




###################################################################################################################################################################################

# Boxplot of kt score

###################################################################################################################################################################################



# In[1] Cycle-tree, no slingshot
direct1 = "../results/"
direct2 = "../data/sim/Symsim/"

kt_cellpath = np.load(file = direct1 + "cellpath/kt_cellpath_cycletree.npy")
kt_cellpath_fast = np.load(file = direct1 + "cellpath/kt_cellpath_cycletree_fast.npy")

kt_slingshot = []
kt_cellrank = []
kt_vetra1 = []
kt_vetra2 = []
kt_vetra3 = []
kt_vdpt = []
for rand in range(1,11):
    adata = anndata.read_h5ad(direct2 + "cycletree_rand" + str(rand) + ".h5ad")
    # vdpt
    scv.tl.velocity_pseudotime(adata)
    pt_vdpt = adata.obs['velocity_pseudotime'].values
    pt_true = adata.obs["sim_time"].values
    kt_vdpt.extend([bmk.kendalltau(pt_vdpt, pt_true)])

    # cellrank
    pt_cellrank = pd.read_csv(direct1 + "cellrank/simulated/cycle_tree/cycletree_rand" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
    pt_cellrank.index = np.arange(pt_cellrank.shape[0])
    for i in range(pt_cellrank.shape[1]):
        pt_i = pt_cellrank.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_cellrank.append(kendalltau(pt_i, true_i))

    # slingshot
    pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/cycle_tree/cycletree_rand" + str(rand) + "_slingshot.tsv", sep = "\t", header = None)

    for i in range(pt_slingshot.shape[1]):
        pt_i = pt_slingshot.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_slingshot.append(kendalltau(pt_i, true_i))
    
    # vetra
    for i in range(1,5):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.4/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.4/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra1.append(kendalltau(pt_i, true_i))

    for i in range(1,5):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra2.append(kendalltau(pt_i, true_i))

    for i in range(1,5):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.8/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.8/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra3.append(kendalltau(pt_i, true_i))

# Plot result
kt_vdpt = np.array(kt_vdpt)
kt_cellpath = np.array(kt_cellpath)
kt_cellrank = np.array(kt_cellrank).reshape(-1)
# kt_cellrank = np.array([1] * kt_cellrank.shape[0])
kt_slingshot = np.array(kt_slingshot)
kt_vetra1 = np.array(kt_vetra1)
kt_vetra2 = np.array(kt_vetra2)
kt_vetra3 = np.array(kt_vetra3)
scores = pd.DataFrame(columns = ["kt", "method"])
scores["kt"] = np.concatenate((kt_cellpath, kt_cellrank, (kt_vetra1 + kt_vetra2 + kt_vetra3)/3, kt_vdpt), axis = 0)
# scores["method"] = ["cellpath(exact)"] * kt_cellpath.shape[0] + ["cellpath(fast)"] * kt_cellpath_fast.shape[0] + ["cellrank"] * kt_cellrank.shape[0] + ["slingshot"] * kt_slingshot.shape[0] + ["vetra"] * kt_vetra.shape[0]
scores["method"] = ["CellPath"] * kt_cellpath.shape[0] + ["CellRank"] * kt_cellrank.shape[0] + ["VeTra"] * kt_vetra1.shape[0] + ["Vdpt"] * kt_vdpt.shape[0]

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = scores, x = "method", y = "kt", ax = ax)
ax.set_xlabel("Method", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
# ax.set_title("cycle-tree", fontsize = 20)
fig.savefig(direct1 + "/plots/cycle-tree.png", bbox_inches = "tight")



# In[3] Multi-cycles, no slingshot
direct1 = "../results/"
direct2 = "../data/sim/Symsim/"
kt_cellpath = np.load(file = direct1 + "cellpath/kt_cellpath_multicycles_80.npy")
kt_cellpath_fast = np.load(file = direct1 + "cellpath/kt_cellpath_multicycles_80_fast.npy")
kt_slingshot = []
kt_cellrank = []
kt_vetra1 = []
kt_vetra2 = []
kt_vetra3 = []
kt_reCAT =[]
kt_vdpt = []
for rand in range(1,6):
    adata = anndata.read_h5ad(direct2 + "multi_cycles_200_rand" + str(rand) + ".h5ad")
    # vdpt
    scv.tl.velocity_pseudotime(adata)
    pt_vdpt = adata.obs['velocity_pseudotime'].values
    pt_true = adata.obs["sim_time"].values
    kt_vdpt.extend([bmk.kendalltau(pt_vdpt, pt_true)])

    # cellrank
    pt_cellrank = pd.read_csv(direct1 + "cellrank/simulated/multi_cycle/multi_cycle_rand" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
    pt_cellrank.index = np.arange(pt_cellrank.shape[0])
    for i in range(pt_cellrank.shape[1]):
        pt_i = pt_cellrank.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_cellrank.append(kendalltau(pt_i, true_i))

    # slingshot
    pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/multi_cycle/multi_cycle_rand" + str(rand) + "_slingshot.tsv", sep = "\t", header = None)

    for i in range(pt_slingshot.shape[1]):
        pt_i = pt_slingshot.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_slingshot.append(kendalltau(pt_i, true_i))
    
    # vetra
    i = 1
    select_cell = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.4/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.4/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    true_i = adata.obs['sim_time'].iloc[select_cell].values
    kt_vetra1.append(kendalltau(pt_i, true_i))

    i = 1
    select_cell = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    true_i = adata.obs['sim_time'].iloc[select_cell].values
    kt_vetra2.append(kendalltau(pt_i, true_i))

    i = 1
    select_cell = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.8/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.8/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    true_i = adata.obs['sim_time'].iloc[select_cell].values
    kt_vetra3.append(kendalltau(pt_i, true_i))

    pt_infer = pd.read_csv(direct1 + "reCAT/pt_200cells_rand"+str(rand)+".csv", index_col = 0, sep = "\t").values.squeeze()
    pt_true = adata.obs["sim_time"].values.squeeze()
    kt = kendalltau(pt_infer[200:], pt_true[200:])
    kt_reCAT.append(kt)

# Plot result
kt_vdpt = np.array(kt_vdpt)
kt_cellpath = np.array(kt_cellpath)
kt_cellrank = np.array(kt_cellrank).reshape(-1)
kt_slingshot = np.array(kt_slingshot)
kt_vetra1 = np.array(kt_vetra1)
kt_vetra2 = np.array(kt_vetra2)
kt_vetra3 = np.array(kt_vetra3)
kt_reCAT = np.array(kt_reCAT)
scores = pd.DataFrame(columns = ["kt", "method"])
scores["kt"] = np.concatenate((kt_cellpath_fast, kt_cellrank, (kt_vetra1 + kt_vetra2 + kt_vetra3)/3, kt_reCAT, kt_vdpt), axis = 0)
scores["method"] = ["CellPath"] * kt_cellpath_fast.shape[0] + ["CellRank"] * kt_cellrank.shape[0] + ["VeTra"] * kt_vetra1.shape[0] + ["reCAT"] * kt_reCAT.shape[0] + ["Vdpt"] * kt_vdpt.shape[0]

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = scores, x = "method", y = "kt", ax = ax)
ax.set_xlabel("Method", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
# ax.set_title("multi-cycle", fontsize = 20)
fig.savefig(direct1 + "plots/multi-cycle.png", bbox_inches = "tight")


# In[4] Dyngen_tree
direct1 = "../results/"
direct2 = "../data/sim/dyngen_tree/"
kt_cellpath = np.load(file = direct1 + "cellpath/kt_cellpath_dyngen_tree_fast.npy")
kt_slingshot = []
kt_cellrank = []
kt_vetra1 = []
kt_vetra2 = []
kt_vetra3 = []
# kt_vetra_dynamical = []
for rand in range(1,5):
    # cellrank, latent time
    adata = anndata.read_h5ad(direct2 + "binary_tree" + str(rand) + ".h5ad")
    # cellrank
    pt_cellrank = pd.read_csv(direct1 + "cellrank/simulated/dyngen_tree/dyngen_tree" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
    pt_cellrank.index = np.arange(pt_cellrank.shape[0])
    for i in range(pt_cellrank.shape[1]):
        pt_i = pt_cellrank.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index]
        kt_cellrank.append(kendalltau(pt_i, true_i))

    # slingshot
    pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/dyngen_tree/dyngen_tree_rand" + str(rand) + "_slingshot.tsv", sep = "\t", header = None)

    for i in range(pt_slingshot.shape[1]):
        pt_i = pt_slingshot.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_slingshot.append(kendalltau(pt_i, true_i))
    
    # vetra
    for i in range(1,4):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.4/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.4/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra1.append(kendalltau(pt_i, true_i))

    for i in range(1,4):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra2.append(kendalltau(pt_i, true_i))
    
    for i in range(1,4):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.8/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.8/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra3.append(kendalltau(pt_i, true_i))



kt_slingshot = np.array(kt_slingshot)
kt_vetra1 = np.array(kt_vetra1)
kt_vetra2 = np.array(kt_vetra2)
kt_vetra3 = np.array(kt_vetra3)
kt_cellrank = np.array(kt_cellrank)
scores = pd.DataFrame(columns = ["kt", "method"])
scores["kt"] = np.concatenate((kt_cellpath, kt_cellrank, kt_slingshot, (kt_vetra1 + kt_vetra2 + kt_vetra3)/3), axis = 0)
scores["method"] = ["CellPath"] * kt_cellpath.shape[0] + ["CellRank"] * kt_cellrank.shape[0] + ["Slingshot"] * kt_slingshot.shape[0] + ["VeTra"] * kt_vetra1.shape[0]


fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = scores, x = "method", y = "kt", ax = ax)
# ax.set_title("dyngen_tree", fontsize = 20)
ax.set_xlabel("Method", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig(direct1 + "/plots/dyngen_tree.png", bbox_inches = "tight")


# In[6] VeloSim-tree
direct1 = "../results/"
direct2 = "../data/sim/velosim_tree/"

kt_cellpath = np.load(file = direct1 + "cellpath/kt_cellpath_velosim_tree_fast.npy")
kt_slingshot = []
kt_cellrank = []
kt_vetra1 = []
kt_vetra2 = []
kt_vetra3 = []
kt_vetra_dynamical = []
datasets = ["3branches_rand0", "3branches_rand1", "3branches_rand3", "3branches_rand4", "4branches_rand0"]
for data in datasets:
    # cellrank, use latent time
    adata = anndata.read_h5ad(direct2 + data + ".h5ad")

    pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/velosim_tree/" + data + "_latent_cellrank.tsv", sep = "\t", index_col = 0)
    pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])
    for i in range(pt_cellrank_latent.shape[1]):
        pt_i = pt_cellrank_latent.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index]
        kt_cellrank.append(kendalltau(pt_i, true_i))

    # slingshot
    pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/velosim_tree/" + data + "_slingshot.tsv", sep = "\t", header = None)

    for i in range(pt_slingshot.shape[1]):
        pt_i = pt_slingshot.iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [x for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt_slingshot.append(kendalltau(pt_i, true_i))
    
    # vetra
    if data != "4branches_rand0":
        counts = 3
    else:
        counts = 4
    for i in range(1,counts):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.4/TI_results_" + data + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.4/TI_results_" + data + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra1.append(kendalltau(pt_i, true_i))

    if data != "4branches_rand0":
        counts = 3
    else:
        counts = 4
    for i in range(1,counts):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.5/TI_results_" + data + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.5/TI_results_" + data + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra2.append(kendalltau(pt_i, true_i))

    if data != "4branches_rand0":
        counts = 3
    else:
        counts = 4
    for i in range(1,counts):
        select_cell = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.8/TI_results_" + data + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
        pt_i = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.8/TI_results_" + data + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
        true_i = adata.obs['sim_time'].iloc[select_cell].values
        kt_vetra3.append(kendalltau(pt_i, true_i))    




kt_cellrank = np.array(kt_cellrank).reshape(-1)
kt_slingshot = np.array(kt_slingshot)
kt_vetra1 = np.array(kt_vetra1)
kt_vetra2 = np.array(kt_vetra2)
kt_vetra3 = np.array(kt_vetra3)
scores = pd.DataFrame(columns = ["kt", "method"])
scores["kt"] = np.concatenate((kt_cellpath, kt_cellrank, kt_slingshot, (kt_vetra1 + kt_vetra2 + kt_vetra3)/3), axis = 0)
scores["method"] = ["CellPath"] * kt_cellpath.shape[0] + ["CellRank"] * kt_cellrank.shape[0] + ["Slingshot"] * kt_slingshot.shape[0] + ["VeTra"] * kt_vetra1.shape[0] 

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
sns.boxplot(data = scores, x = "method", y = "kt", ax = ax)
# ax.set_title("velosim_tree", fontsize = 20)
ax.set_xlabel("Method", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 20)
fig.savefig(direct1 + "plots/velosim_tree.png", bbox_inches = "tight")



###################################################################################################################################################################################

# Plot figures

###################################################################################################################################################################################
# In[8] Plot dyngen tree
def plot_pt(adata, pseudo_order, basis = "pca", trajs = 4, figsize= (20,20), save_as = None, title = None, axis = True):

    if trajs >= 2:
        nrows = np.ceil(trajs/2).astype('int32')
        ncols = 2

    elif trajs == 1:
        nrows = 1
        ncols = 1        
        
    else:
        raise ValueError("invalid trajectory numbers")

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    if title:
        fig.suptitle("first order approximate pseudo-time", fontsize = 18)

    basis = 'X_' + basis
    if basis not in adata.obsm.keys():
        raise ValueError("basis incorrect")


    for i in range(min(trajs, pseudo_order.shape[1])):
        sorted_pt = pseudo_order["traj_"+str(i)].dropna(axis = 0).sort_values()
        # traj = [int(x.split("_")[1]) for x in sorted_pt.index]
        # X_traj = adata.obsm[basis][traj,:]
        X_traj = adata[sorted_pt.index.values,:].obsm[basis]
 
        if nrows != 1:
            # multiple >2 plots
            if not axis:
                axs[i%nrows, i//nrows].axis("off")
            axs[i%nrows, i//nrows].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            

            pseudo_visual = axs[i%nrows, i//nrows].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)

            axs[i%nrows, i//nrows].set_title("Path " + str(i), fontsize = 25)
            if basis.split("_")[1] == "pca":
                prefix = "PC"
            else:
                prefix = basis.split("_")[1]
            axs[i%nrows, i//nrows].set_xlabel(prefix + " 1", fontsize = 19)
            axs[i%nrows, i//nrows].set_ylabel(prefix + " 2", fontsize = 19)

            axs[i%nrows, i//nrows].set_xticks([])
            axs[i%nrows, i//nrows].set_yticks([])
            axs[i%nrows, i//nrows].spines['right'].set_visible(False)
            axs[i%nrows, i//nrows].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i%nrows, i//nrows])
            cbar.ax.tick_params(labelsize = 20)

        elif nrows == 1 and ncols == 1:
            # one plot
            if not axis:
                axs.axis("off")            
            axs.scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)

            pseudo_visual = axs.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)

            axs.set_title("Path " + str(i), fontsize = 25)
            if basis.split("_")[1] == "pca":
                prefix = "PC"
            else:
                prefix = basis.split("_")[1]
            axs.set_xlabel(prefix + " 1", fontsize = 19)
            axs.set_ylabel(prefix + " 2", fontsize = 19)
            axs.set_xticks([])
            axs.set_yticks([])
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)   

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs)
            cbar.ax.tick_params(labelsize = 20)

        else:
            # two plots
            if not axis:
                axs[i].axis("off")
            axs[i].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            pseudo_visual = axs[i].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)
            
            axs[i].set_title("Path " + str(i), fontsize = 25)
            if basis.split("_")[1] == "pca":
                prefix = "PC"
            else:
                prefix = basis.split("_")[1]
            axs[i].set_xlabel(prefix + " 1", fontsize = 19)
            axs[i].set_ylabel(prefix + " 2", fontsize = 19)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i])
            cbar.ax.tick_params(labelsize = 20)

 
    
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()    

direct1 = "../results/"
direct2 = "../data/sim/dyngen_tree/"
use_dynamical = False
num_trajs = 3
num_metacells = 200
kt_cellpath = []
rand = 2
adata = anndata.read_h5ad(direct2 + "binary_tree" + str(rand) + ".h5ad")
if use_dynamical:
    scv.tl.recover_dynamics(adata, n_jobs = 12)
    scv.tl.velocity(adata, mode = "dynamical")
    gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
    adata = adata[:,gene_idx]
else:
    scv.tl.velocity(adata, mode = "stochastic")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis = "umap", arrow_length = 3, color = "sim_time")
# scv.pl.velocity_embedding_stream(adata, basis = "umap", color = "sim_time")

cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)

plot_pt(cellpath_obj.adata, cellpath_obj.pseudo_order, basis = "umap", trajs = num_trajs, figsize=(20,14), save_as= direct1 + "plots/dyngen_tree_cellpath.png")


# cellrank
rand = 2
adata = anndata.read_h5ad(direct2 + "binary_tree" + str(rand) + ".h5ad")
pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/dyngen_tree/dyngen_tree" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])

plot_pt(adata, pt_cellrank_latent, basis = "umap", trajs = 1, figsize=(10,7), save_as= direct1 + "plots/dyngen_tree_cellrank.png")

# slingshot
rand = 2
pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/dyngen_tree/dyngen_tree_rand" + str(rand) + "_slingshot.tsv", sep = "\t", header = None)
pt_slingshot.columns = ["traj_" + str(i) for i in range(2)]

plot_pt(adata,pt_slingshot, basis = "umap", trajs = 2, figsize=(20,7), save_as= direct1 + "plots/dyngen_tree_slingshot.png")


# vetra
pt_vetra = pd.DataFrame(data = np.nan, index = pt_slingshot.index, columns = ["traj_" + str(i) for i in range(4)])
for i in range(1,4):
    select_cell = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/dyngen_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    true_i = adata.obs['sim_time'].iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), "traj_" + str(i - 1)] = pt_i.squeeze()

plot_pt(adata, pt_vetra, basis = "umap", trajs = 4, figsize=(20,14), save_as= direct1 + "plots/dyngen_tree_vetra.png")



# In[9] Plot cycle-tree

num_metacells = 300
num_trajs = 4
kt_dpt = []
kt_cellpath = []
kt_slingshot = []

direct1 = "../results/"
direct2 = "../data/sim/Symsim/"
adata = anndata.read_h5ad(direct2 + "cycletree_rand2.h5ad")
rand = 2

# cellpath
cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
cellpath_obj.meta_cell_construction(flavor = "k-means", n_clusters = 400)
cellpath_obj.meta_cell_graph(k_neighs = 13, pruning = True, distance_scalar = 1)
cellpath_obj.meta_paths_finding(threshold = 0.5, cutoff_length = 5, length_bias = 0.7, mode = "fast")
cellpath_obj.first_order_pt(num_trajs = num_trajs, prop_insert = 0.700)

plot_pt(cellpath_obj.adata, cellpath_obj.pseudo_order, basis = "umap", trajs = num_trajs, figsize=(20,10), save_as= direct1 + "plots/cycle_tree_cellpath.png")


# cellrank
pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/cycle_tree/cycletree_rand" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])
pt_cellrank_latent.columns = ["traj_" + str(i) for i in range(4)]
pts = []
for i in range(pt_cellrank_latent.shape[0]):
    pt = [x for x in pt_cellrank_latent.iloc[i,:].squeeze() if not np.isnan(x)]
    assert len(pt) == 1
    pts.append(pt[0])
pt_cellrank_latent["traj_0"] = np.array(pts)
pt_cellrank_latent = pt_cellrank_latent.loc[:, ["traj_0"]]

plot_pt(adata, pt_cellrank_latent, basis = "umap", trajs = 1, figsize=(10,5), save_as= direct1 + "plots/cycle_tree_cellrank.png")

# vetra
pt_vetra = pd.DataFrame(data = np.nan, index = pt_cellrank_latent.index, columns = ["traj_" + str(i) for i in range(4)])
for i in range(1,5):
    select_cell = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/cycle_tree/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), "traj_" + str(i - 1)] = pt_i.squeeze()

plot_pt(adata, pt_vetra, basis = "umap", trajs = 4, figsize=(20,14), save_as= direct1 + "plots/cycle_tree_vetra.png")


# vdpt
# cellrank
adata = anndata.read_h5ad(direct2 + "cycletree_rand2.h5ad")
scv.tl.velocity_pseudotime(adata)
pt_vdpt = adata.obs.loc[:, ['velocity_pseudotime']]
pt_vdpt = pt_vdpt.rename(columns={'velocity_pseudotime': 'traj_0'})

plot_pt(adata, pt_vdpt, basis = "umap", trajs = 1, figsize=(10,5), save_as= direct1 + "plots/cycle_tree_vdpt.png")


# In[16] Plot Multi-cycles
# cellpath
num_metacells = 80
num_trajs = 1
direct1 = "../results/"
direct2 = "../data/sim/Symsim/"

rand = 1
adata = anndata.read_h5ad(direct2 + "multi_cycles_200_rand" + str(rand) + "_clust.h5ad")

# cellpath
cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
# here we use fast implementation, the flavor can also be changed to "k-means" for k-means clustering
# cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = None, n_neighs = 5, num_trajs = num_trajs, insertion = True, prop_insert = 0.30, seed = 0, mode = "exact", pruning = False, scaling = 4, distance_scalar = 0.5)
cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = None, n_neighs = 5, num_trajs = num_trajs, insertion = True, prop_insert = 0.70, seed = 0, mode = "fast", pruning = False, scaling = 4, distance_scalar = 0.5)
# plot_pt(cellpath_obj.adata, cellpath_obj.pseudo_order, basis = "pca", trajs = num_trajs, figsize=(10,5), save_as= direct1 + "plots/multicycles_cellpath.png")

fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot()

ax.scatter(cellpath_obj.adata.obsm["X_pca"][:,0],cellpath_obj.adata.obsm["X_pca"][:,1], color = 'gray', alpha = 0.1)

for i in range(1):
    sorted_pt = cellpath_obj.pseudo_order["traj_"+str(i)].dropna(axis = 0).sort_values()
    # traj = [int(x.split("_")[1]) for x in sorted_pt.index]
    # X_traj = adata.obsm[basis][traj,:]
    X_traj = cellpath_obj.adata[sorted_pt.index,:].obsm["X_pca"]
    for i in range(X_traj.shape[0]-1):
        line = ax.plot(X_traj[i:(i+2), 0], X_traj[i:(i+2), 1], 'gray', '-', alpha = 1)
        visual.add_arrow(line[0], size = 10)

    pic = ax.scatter(X_traj[:,0],X_traj[:,1], c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)
ax.tick_params(axis = "both", direction = "out", labelsize = 16)
ax.set_xlabel("PC1", fontsize = 19)
ax.set_ylabel("PC2", fontsize = 19)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("CellPath: Path " + str(0), fontsize = 25)
fig.savefig(direct1 + "plots/multicycles_cellpath.png", bbox_inches = "tight")

# cellrank
pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/multi_cycle/multi_cycle_rand" + str(rand) + "_cellrank.tsv", sep = "\t", index_col = 0)
pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])
pt_cellrank_latent.columns = ["traj_" + str(i) for i in range(1)]
pts = []
for i in range(pt_cellrank_latent.shape[0]):
    pt = pt_cellrank_latent.iloc[i,0]
    pts.append(pt)
pt_cellrank_latent["traj_0"] = np.array(pts)
pt_cellrank_latent = pt_cellrank_latent.loc[:, ["traj_0"]]
pt_cellrank_latent
plot_pt(adata, pt_cellrank_latent, basis = "pca", trajs = 1, figsize=(10,5), save_as = direct1 + "plots/multicycles_cellrank.png")

# vetra
pt_vetra = pd.DataFrame(data = np.nan, index = pt_cellrank_latent.index, columns = ["traj_" + str(i) for i in range(0)])
i = 1
select_cell = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.5/TI_results_rand" + str(rand) + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
pt_i = pd.read_csv(direct1 + "vetra/simulated/multi_cycle/min_dist_0.5/TI_results_rand" + str(rand) + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
true_i = adata.obs['sim_time'].iloc[select_cell].values
pt_vetra.loc[select_cell.squeeze(), "traj_" + str(i - 1)] = pt_i.squeeze()
plot_pt(adata, pt_vetra, basis = "pca", trajs = 1, figsize=(10,5), save_as = direct1 + "plots/multicycles_vetra.png")


# In[17] VeloSim tree

direct1 = "../results/"
direct2 = "../data/sim/velosim_tree/"
num_metacells = 200
kt_cellpath = []
num_trajs = 3

# cellpath
dataset = "3branches_rand3"
adata = anndata.read_h5ad(direct2 + dataset + ".h5ad")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis = "pca", arrow_length = 3, color = "sim_time")
X_pca = adata.obsm["X_pca"]
pt_true = adata.obs["sim_time"].values

cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, n_neighs = 15, num_trajs = num_trajs, prop_insert = 0.70, seed = 0, mode = "fast", pruning = True, scaling = 4, distance_scalar = 0.5)
cellpath_obj.adata.obsm["X_pca"] = X_pca
plot_pt(cellpath_obj.adata, cellpath_obj.pseudo_order, basis="pca", trajs = num_trajs, figsize=(20,14), save_as= direct1 + "plots/velosim_tree_cellpath.png")

# slingshot
pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/velosim_tree/" + dataset + "_slingshot.tsv", sep = "\t", header = None)
pt_slingshot.columns = ["traj_" + str(i) for i in range(3)]

plot_pt(adata, pt_slingshot, basis = "pca", trajs = 3, figsize=(20,14), save_as= direct1 + "plots/velosim_tree_slingshot.png")

# cellrank
pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/velosim_tree/" + dataset + "_latent_cellrank.tsv", sep = "\t", index_col = 0)
pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])
pt_cellrank_latent.columns = ["traj_" + str(i) for i in range(1)]
pts = []
for i in range(pt_cellrank_latent.shape[0]):
    # pt = [x for x in pt_cellrank_latent.iloc[i,:].squeeze() if not np.isnan(x)]
    pt = [pt_cellrank_latent.iloc[i,0]]
    # assert len(pt) == 1
    pts.append(pt[0])
pt_cellrank_latent["traj_0"] = np.array(pts)
pt_cellrank_latent = pt_cellrank_latent.loc[:, ["traj_0"]]
plot_pt(adata, pt_cellrank_latent, basis = "pca", trajs = 1, figsize=(10,7), save_as = direct1 + "plots/velosim_tree_cellrank.png")

# vetra
pt_vetra = pd.DataFrame(data = np.nan, index = pt_slingshot.index, columns = ["traj_" + str(i) for i in range(4)])
counts = 4

for i in range(1,counts):
    select_cell = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.5/TI_results_" + dataset + "/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/velosim_tree/min_dist_0.5/TI_results_" + dataset + "/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), "traj_" + str(i - 1)] = pt_i.squeeze()

plot_pt(adata, pt_vetra, basis = "pca", trajs = 3, figsize=(20,14), save_as= direct1 + "plots/velosim_tree_vetra.png")

# In[] Plot bifur
num_metacells = 500
num_trajs = 4

direct1 = "../results/"
direct2 = "../data/sim/"

# cellpath
adata = anndata.read_h5ad(direct2 + "Bifurcating.h5ad")
cellpath_obj = cp.CellPath(adata = adata)
cellpath_obj.meta_cell_construction(n_clusters = num_metacells)
cellpath_obj.meta_cell_graph(k_neighs = 10, pruning = False)
cellpath_obj.meta_paths_finding(threshold = 0.7, cutoff_length = None, length_bias = 0.7, max_trajs = 30)
cellpath_obj.first_order_pt(num_trajs = num_trajs, prop_insert = 0.7)
visual.first_order_approx_pt(cellpath_obj, basis = "pca", trajs = num_trajs, figsize = (20,14), save_as = direct1 + "plots/bifur_cellpath.png")

# slingshot
pt_slingshot = pd.read_csv(direct1 + "slingshot/simulated/trifur/Bifurcating_slingshot.tsv", sep = "\t", header = None)
pt_slingshot.columns = ["traj_" + str(i) for i in range(2)]
plot_pt(adata, pt_slingshot, basis = "pca", trajs = 2, figsize=(20,7), save_as= direct1 + "plots/bifur_slingshot.png")

# cellrank
pt_cellrank_latent = pd.read_csv(direct1 + "cellrank/simulated/bifur/bifur_cellrank.tsv", sep = "\t", index_col = 0)
pt_cellrank_latent.index = np.arange(pt_cellrank_latent.shape[0])
pt_cellrank_latent.columns = ["traj_" + str(i) for i in range(1)]
pts = []
for i in range(pt_cellrank_latent.shape[0]):
    # pt = [x for x in pt_cellrank_latent.iloc[i,:].squeeze() if not np.isnan(x)]
    pt = [pt_cellrank_latent.iloc[i,0]]
    # assert len(pt) == 1
    pts.append(pt[0])
pt_cellrank_latent["traj_0"] = np.array(pts)
pt_cellrank_latent = pt_cellrank_latent.loc[:, ["traj_0"]]
plot_pt(cellpath_obj.adata, pt_cellrank_latent, basis = "pca", trajs = 1, figsize=(10,7), save_as = direct1 + "plots/bifur_cellrank.png")

# vetra
pt_vetra = pd.DataFrame(data = np.nan, index = pt_cellrank_latent.index, columns = ["traj_" + str(i) for i in range(5)])
counts = 5

for i in range(1,counts):
    select_cell = pd.read_csv(direct1 + "vetra/simulated/bifur/TI_results/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv(direct1 + "vetra/simulated/bifur/TI_results/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), "traj_" + str(i - 1)] = pt_i.squeeze()

plot_pt(cellpath_obj.adata, pt_vetra, basis = "pca", trajs = 4, figsize=(20,14), save_as= direct1 + "plots/bifur_vetra.png")


# %%
