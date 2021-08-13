# In[0]
import sys
sys.path.append("../CellPath/")

import scvelo as scv
import numpy as np  
import matplotlib.pyplot as plt

import anndata

import pandas as pd
import scprep

import sys
sys.path.append("..")


import cellpath as cp
import cellpath.visual as visual
import cellpath.benchmark as bmk 
import cellpath.de_analy as de


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# In[1]
num_trajs = 4
include_all_cells = True
num_metacells = 500
resolution = 35

PATH = "../CellPath/sim_data/Dyngen/"
adata = anndata.read_h5ad(PATH + "Trifurcating.h5ad")

adata = adata[adata.obs["simulation_i"] != 4,:]
orig = adata.obs["simulation_i"].values
orig = np.where(orig == 1, 0, orig)
orig = np.where(orig == 2, 1, orig)
orig = np.where(orig == 3, 2, orig)
adata.obs["simulation_i"] = np.where(orig == 5, 3, orig)
adata.obs["simulation_i"] = adata.obs["simulation_i"].astype("category")
adata.obs["clusters"] = ["ground truth traj " + str(x) for x in adata.obs["simulation_i"].values]
adata.write_h5ad(PATH + "Bifurcating.h5ad")

cellpath_obj = cp.CellPath(adata = adata)
cellpath_obj.meta_cell_construction(n_clusters = num_metacells)
cellpath_obj.meta_cell_graph(k_neighs = 10, pruning = False)
cellpath_obj.meta_paths_finding(threshold = 0.7, cutoff_length = None, length_bias = 0.7, max_trajs = 30)
cellpath_obj.first_order_pt(num_trajs = num_trajs, prop_insert = 1.0)

# cellpath_obj.all_in_one(flavor = "k-means", num_metacells = num_metacells, resolution = resolution, n_neighs = 10, pruning = False, num_trajs = num_trajs, mode = "fast", length_bias = 0.7)

pca_op = PCA(n_components = 2)
cellpath_obj.adata.obsm["X_pca"] = pca_op.fit_transform(StandardScaler().fit_transform(adata.X.todense()))

visual.meta_traj_visual(cellpath_obj, basis = "pca", trajs = num_trajs, figsize = (12,7), save_as = None, legend_pos = "upper left", axis = True, arrow_size = 20, linewidth = 2, markerscale = 2.0)
visual.first_order_approx_pt(cellpath_obj, basis = "pca", trajs = num_trajs, figsize = (20,14), save_as = None)



# In[]
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

# average entropy score of cellpath
kt = bmk.cellpath_kt(cellpath_obj)

# average entropy score of slingshot
kt_slingshot = []
pt_slingshot = pd.read_csv("slingshot/simulated/trifur/Bifurcating_slingshot.tsv", sep = "\t", header = None)
pt_slingshot.index = cellpath_obj.adata.obs.index.values

for i in range(pt_slingshot.shape[1]):
    pt_i = pt_slingshot.iloc[:,i]
    pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
    ordering = [x for x in pt_i[pt_index].sort_values().index]
    pt_i = pt_i[pt_index]
    true_i = adata.obs['sim_time'].loc[pt_index].values
    kt_slingshot.append(kendalltau(pt_i, true_i))

kt_slingshot = np.array(kt_slingshot)

# average entropy score of vetra
pt_vetra = pd.DataFrame(data = np.nan, index = cellpath_obj.adata.obs.index.values, columns = [i for i in range(4)])
kt_vetra = []
counts = 5
for i in range(1,counts):
    select_cell = pd.read_csv("vetra/simulated/bifur/TI_results/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv("vetra/simulated/bifur/TI_results/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), i-1] = pt_i.squeeze()

    true_i = adata.obs['sim_time'].iloc[select_cell].values
    kt_vetra.append(kendalltau(pt_i, true_i))

kt_vetra = np.array(kt_vetra)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot()

ax.scatter([0 for x in kt.values()], [x for x in kt.values()])
ax.scatter([1 for x in kt_slingshot], [x for x in kt_slingshot])
ax.scatter([2 for x in kt_vetra], [x for x in kt_vetra])

ax.set_title("Branching dataset", fontsize = 20)
ax.set_ylabel("kendall tau", fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks([-0.5,0,1,2,2.5],fontsize = 15)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = ""
labels[1] = "CellPaths"
labels[2] = "Slingshot"
labels[3] = "VeTra"
labels[4] = ""
ax.set_xticklabels(labels)
fig.savefig("./plots/trifur_kt.png", bbox_inches = "tight")
# In[2]

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))

adata = cellpath_obj.adata
basis = 'X_pca'
if basis not in adata.obsm.keys():
    raise ValueError("basis incorrect")

# sorted_pt = cellpath_obj.adata.obs["sim_time"].dropna(axis = 0).sort_values()
sorted_pt = cellpath_obj.adata.obs["sim_time"].loc[cellpath_obj.adata.obs["sim_time"] > 120].dropna(axis = 0).sort_values()
X_traj = adata[sorted_pt.index,:].obsm[basis]

# one plot         
axs.scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)

pseudo_visual = axs.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)

axs.set_title("sim_time", fontsize = 25)
axs.set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
axs.set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
axs.set_xticks([])
axs.set_yticks([])
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)   

cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs)
cbar.ax.tick_params(labelsize = 20)

plt.show()     

# In[2] Find branching point

def plot_pt(adata, pseudo_order, basis = "pca", trajs = 4, figsize= (20,20), save_as = None, title = None, axis = True):

    if trajs >= 2:
        nrows = np.ceil(trajs/2).astype('int32')
        ncols = 2
        # nrows = 4
        # ncols = 1

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
        sorted_pt = pseudo_order.iloc[:,i].dropna(axis = 0).sort_values()
        # traj = [int(x.split("_")[1]) for x in sorted_pt.index]
        # X_traj = adata.obsm[basis][traj,:]
        X_traj = adata[sorted_pt.index.values,:].obsm[basis]
 
        if nrows != 1:
            # multiple >2 plots
            if not axis:
                axs[i%nrows, i//nrows].axis("off")
            axs[i%nrows, i//nrows].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            

            pseudo_visual = axs[i%nrows, i//nrows].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)

            axs[i%nrows, i//nrows].set_title("VeTra: Path " + str(i), fontsize = 25)
            # axs[i%nrows, i//nrows].set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            # axs[i%nrows, i//nrows].set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs[i%nrows, i//nrows].set_xlabel("PC 1", fontsize = 19)
            axs[i%nrows, i//nrows].set_ylabel("PC 2", fontsize = 19)

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

            axs.set_title("VeTra: Path " + str(i), fontsize = 25)
            # axs.set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            # axs.set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs.set_xlabel("PC 1", fontsize = 19)
            axs.set_ylabel("PC 2", fontsize = 19)
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
            
            axs[i].set_title("VeTra: Path " + str(i), fontsize = 25)
            # axs[i].set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            # axs[i].set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs[i].set_xlabel("PC 1", fontsize = 19)
            axs[i].set_ylabel("PC 2", fontsize = 19)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i])
            cbar.ax.tick_params(labelsize = 20)

 
    
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()    


def plot_data(cellpath_obj, basis = "umap", figsize = (15,7), save_as = None, title = None, **kwargs):
    """\
    Description
        Plot original dataset
    
    Parameters
    ----------
    cellpath_obj
        cellpath object
    basis
        the basis used for visualization
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    """

    _kwargs = {
        "axis": False,
        "legend_pos": "upper left",
        "colormap": "tab20",
        "s": 10,
        "add_arrow": False,
        "markerscale": 3.0
    }
    _kwargs.update(kwargs)

    X = cellpath_obj.adata.obsm["X_" + basis]
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()

    ax.tick_params(axis = "both", direction = "out", labelsize = 16)
    ax.set_xlabel("PC 1 (explained variance 41.69%)", fontsize = 19)
    ax.set_ylabel("PC 2 (explained variance 26.42%)", fontsize = 19)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    cluster_anno = [x for x in cellpath_obj.adata.obs["clusters"].values]
    cluster_uni = [x for x in np.unique(cellpath_obj.adata.obs["clusters"].values)]

    colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_uni))

    for count, clust in enumerate(cluster_uni):
        idx = np.where(np.array(cluster_anno) == clust)[0]
        ax.scatter(X[idx,0], X[idx,1], color = colormap(count), alpha = 0.7, label = clust, s = _kwargs["s"])    
    ax.legend(loc=_kwargs["legend_pos"], prop={'size': 15}, frameon = False, ncol = 1, markerscale=_kwargs["markerscale"])

    X_highlight = np.argmin(np.abs(cellpath_obj.adata.obs["sim_time"].values - 125))
    print(X_highlight)

    ax.scatter(X[X_highlight,0], X[X_highlight,1], color = "black", alpha = 1, s = 50)
    
    if title is not None:
        ax.set_title(title, fontsize = 25)
    if save_as != None:
        fig.savefig(save_as, bbox_inches = 'tight')


# find branching point
simtime = cellpath_obj.adata.obs["sim_time"].values
cuts = np.quantile(a = simtime, q = np.arange(start = 0, stop = 1, step = 0.05))
cuts = np.append(cuts, np.max(simtime))
for i_cut, cut in enumerate(cuts):
    if i_cut == 0:
        continue
    else:
        prev_cut = cuts[i_cut - 1]
    adata_cut = cellpath_obj.adata[(cellpath_obj.adata.obs["sim_time"] < cuts[i_cut]).values & (cellpath_obj.adata.obs["sim_time"] > prev_cut).values]
    mu = np.mean((adata_cut.X).todense(), axis = 0)
    var = np.mean(np.array(adata_cut.X.todense() - mu) ** 2)
    print(var)
    if var > 0.13:
        break
print(cuts[i_cut-1])
# HERE IS 125
plot_data(cellpath_obj, basis = "pca", figsize = (12,7))


# In[] Average entropy score
cut = 138
# average entropy score of cellpath
kt = bmk.cellpath_kt(cellpath_obj)

# plot_pt(adata, cellpath_obj.pseudo_order, basis = "pca", trajs = 4, figsize=(20,14), save_as= "./plots/cellpath_trifur.png")

cellpath_obj.pseudo_order = cellpath_obj.pseudo_order[cellpath_obj.adata.obs["sim_time"] > cut]

plot_pt(adata, cellpath_obj.pseudo_order, basis = "pca", trajs = 4, figsize=(20,14), save_as= None)

bmk_belongings_cellpath = bmk.purity_count(cellpath_obj = cellpath_obj, method = "CellPath", trajs = num_trajs)
entro_cellpath = bmk.average_entropy(bmk_belongings_cellpath)

# In[3]
# average entropy score of slingshot
pt_slingshot = pd.read_csv("slingshot/simulated/trifur/Bifurcating_slingshot.tsv", sep = "\t", header = None)
pt_slingshot.index = cellpath_obj.adata.obs.index.values

# plot_pt(adata, pt_slingshot, basis = "pca", trajs = 2, figsize=(20,7), save_as= "./plots/slingshot_trifur.png")

pt_slingshot = pt_slingshot[cellpath_obj.adata.obs["sim_time"] > cut]

plot_pt(adata, pt_slingshot, basis = "pca", trajs = 2, figsize=(20,7), save_as= None)

bmk_belongings_slingshot = bmk.purity_count(adata = adata, method = "Slingshot", slingshot_result = pt_slingshot, trajs = 3)
entro_slingshot = bmk.average_entropy(bmk_belongings_slingshot)

# In[4]
# average entropy score of vetra
pt_vetra = pd.DataFrame(data = np.nan, index = cellpath_obj.adata.obs.index.values, columns = [i for i in range(4)])
counts = 5
for i in range(1,counts):
    select_cell = pd.read_csv("vetra/simulated/bifur/TI_results/cell_select_" + str(i) + ".txt", header = None).values.astype(np.bool).squeeze()
    pt_i = pd.read_csv("vetra/simulated/bifur/TI_results/trajectory_" + str(i) + ".txt", header = None).iloc[select_cell].values
    pt_vetra.loc[select_cell.squeeze(), i-1] = pt_i.squeeze()

# plot_pt(adata, pt_vetra, basis = "pca", trajs = 4, figsize=(20,14), save_as= "./plots/vetra_trifur.png")

pt_vetra = pt_vetra[cellpath_obj.adata.obs["sim_time"] > cut]

plot_pt(adata, pt_vetra, basis = "pca", trajs = 4, figsize=(20,14), save_as= None)

bmk_belongings_vetra = bmk.purity_count(adata = adata, method = "Slingshot", slingshot_result = pt_vetra, trajs = 4)
entro_vetra = bmk.average_entropy(bmk_belongings_vetra)

# plot average entropy score
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
entro_score = [entro_cellpath, entro_slingshot, entro_vetra]
# entro_score = [1.278, 1.344]
ax.barh(["CellPath", "Slingshot", "VeTra"], entro_score, height = 0.3 )

# for index, value in enumerate(entro_score):
#     plt.text(value, index, str(value), fontsize = 16)

ax.tick_params(axis = "both", direction = "out", labelsize = 16)
ax.set_xlabel("average entropy score", fontsize = 19)
# ax.set_ylabel(basis + " 2", fontsize = 19)
# ax.set_yticks([0.1, 0.3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig("./plots/trifur_entropy.png", bbox_inches = "tight")

# %%
