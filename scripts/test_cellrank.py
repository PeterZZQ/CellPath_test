# In[0] 
import numpy as np
import scvelo as scv
import anndata
import cellrank as cr
import pandas as pd
import warnings
import scanpy as sc

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

scv.settings.verbosity = 3
scv.settings.set_figure_params("scvelo")
cr.settings.verbosity = 2

# In[1]
use_dynamic = False

# cycle-tree
rand = 10
path = "../CellPath/sim_data/Symsim/"
path2 = "./cellrank/simulated/cycle_tree/"


adata = anndata.read_h5ad(path + "cycletree_rand" + str(rand) + ".h5ad")
if use_dynamic:
    scv.tl.recover_dynamics(adata, n_jobs = 6)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4)
    
# In[2]
cr.tl.terminal_states(adata, cluster_key="pop", weight_connectivities=0.2)
cr.pl.terminal_states(adata)

# terminal_idx = [idx for idx, x in enumerate(adata.obs["terminal_states"].values) if x is not np.nan]
# terminal_val = [x for idx, x in enumerate(adata.obs["terminal_states"].values) if x is not np.nan]
# terminal_clust = np.unique(np.array(terminal_val))



# In[3]

cr.tl.initial_states(adata, cluster_key="pop")
cr.pl.initial_states(adata, discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

# In[4]
# Lineage

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

"""
sc.tl.leiden(adata)

# how to extract lineages from data, in addition, use the ground truth cluster (update)
scv.tl.paga(
    adata,
    # use leiden instead of pop
    groups="leiden",
    root_key="initial_states_probs",
    end_key="terminal_states_probs",
    # use_time_prior="velocity_pseudotime",
)

cr.pl.cluster_fates(
    adata,
    mode="paga_pie",
    # use leiden instead of pop
    cluster_key="leiden",
    basis="umap",
    legend_kwargs={"loc": "top right out"},
    legend_loc="top left out",
    node_size_scale=5,
    edge_width_scale=1,
    max_edge_width=4,
    title="directed PAGA",
)
"""

# In[5]
if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )


# In[6]
pt = adata.obs["dpt_pseudotime"].values
pt_final = pd.DataFrame(data = np.nan, index = adata.obs.index, columns=lineage_name)
for i, x in enumerate(max_lineage):
    pt_final.iloc[i,x] = pt[i]

pt_final.to_csv(path2 + "cycletree_rand" + str(rand) + "_cellrank.tsv", sep = "\t", na_rep='NA')



# In[7]

use_dynamic = False

# Multi-cycles
rand = 5
path = "../CellPath/sim_data/Symsim/"
path2 = "./cellrank/simulated/multi_cycle/"


adata = anndata.read_h5ad(path + "multi_cycles_200_rand" + str(rand) + ".h5ad")
if use_dynamic:
    scv.tl.recover_dynamics(adata, n_jobs = 6)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4)

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata)

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

pt = adata.obs["dpt_pseudotime"].values
pt_final = pd.DataFrame(data = np.nan, index = adata.obs.index, columns=lineage_name)
for i, x in enumerate(max_lineage):
    pt_final.iloc[i,x] = pt[i]

pt_final.to_csv(path2 + "multi_cycles_rand" + str(rand) + "_cellrank.tsv", sep = "\t", na_rep='NA')



# In[8] Bifur

use_dynamic = True

path = "../CellPath/sim_data/Dyngen/"
path2 = "./cellrank/simulated/bifur/"


adata = anndata.read_h5ad(path + "Bifurcating.h5ad")
if use_dynamic:
    scv.tl.recover_dynamics(adata, n_jobs = 6)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4)

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata)

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
    pt = adata.obs["latent_time"]
else:

    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

    pt = adata.obs["dpt_pseudotime"].values
pt_final = pd.DataFrame(data = np.nan, index = adata.obs.index, columns=lineage_name)
for i, x in enumerate(max_lineage):
    pt_final.iloc[i,x] = pt[i]

pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
pt_final.to_csv(path2 + "bifur_cellrank.tsv", sep = "\t", na_rep='NA')

# In[9] Dyngen tree

use_dynamic = True

path = "./additional_data/dyngen_tree/"
path2 = "./cellrank/simulated/dyngen_tree/"

for rand in range(1, 5):
    adata = anndata.read_h5ad(path + "binary_tree" + str(rand) + ".h5ad")
    # adata.layers["unspliced"] = adata.layers["counts_unspliced"]
    # adata.layers["spliced"] = adata.layers["counts_spliced"]
    # scv.pp.normalize_per_cell(adata)
    # scv.pp.log1p(adata)
    # adata.write_h5ad(path + "binary_tree" + str(rand) + ".h5ad")

    if use_dynamic:
        scv.tl.recover_dynamics(adata, n_jobs = 6)
        scv.tl.velocity(adata, mode="dynamical")
        # scv.tl.velocity_graph(adata)
        # scv.pl.velocity_embedding_stream(adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4)
    else:
        scv.tl.velocity(adata)

    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis = "pca")
    scv.tl.velocity_embedding(adata, basis = "umap")

    cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
    cr.pl.terminal_states(adata)

    cr.tl.initial_states(adata, cluster_key=None)
    cr.pl.initial_states(adata, discrete=True)

    initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
    initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

    cr.tl.lineages(adata)
    cr.pl.lineages(adata, same_plot=False)
    lineage = adata.obsm['to_terminal_states']
    lineage_name = [x for x in lineage.names]
    lineage = np.array(lineage)
    max_lineage = np.argmax(lineage, axis = 1)
    max_lineage = np.array([x for x in max_lineage])

    if use_dynamic:
        scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
        scv.pl.scatter(adata,
            color=["latent_time"],
            fontsize=16,
            cmap="viridis",
            perc=[2, 98],
            colorbar=True,
            rescale_color=[0, 1],
            title=["latent time"],
        )
        pt = adata.obs["latent_time"]
    else:
        
        root_idx = initial_idx[0]
        adata.uns["iroot"] = root_idx
        sc.tl.dpt(adata)
        scv.pl.scatter(adata,
            color=["dpt_pseudotime"],
            fontsize=16,
            cmap="viridis",
            perc=[2, 98],
            colorbar=True,
            rescale_color=[0, 1],
            title=["dpt pseudotime"],
        )

        pt = adata.obs["dpt_pseudotime"].values
    pt_final = pd.DataFrame(data = np.nan, index = adata.obs.index, columns=lineage_name)
    for i, x in enumerate(max_lineage):
        pt_final.iloc[i,x] = pt[i]
    pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
    pt_final.to_csv(path2 + "dyngen_tree" + str(rand) + "_cellrank.tsv", sep = "\t", na_rep='NA')


# In[9] VeloSim tree

use_dynamic = True

path = "./additional_data/velosim_tree/"
path2 = "./cellrank/simulated/velosim_tree/"
datasets = ["3branches_rand0", "3branches_rand1", "3branches_rand3", "3branches_rand4", "4branches_rand0"]
for data in datasets:
    adata = anndata.read_h5ad(path + data + ".h5ad")
    # adata.layers["unspliced"] = adata.layers["counts_unspliced"]
    # adata.layers["spliced"] = adata.layers["counts_spliced"]
    # scv.pp.normalize_per_cell(adata)
    # scv.pp.log1p(adata)
    # adata.write_h5ad(path + "binary_tree" + str(rand) + ".h5ad")

    if use_dynamic:
        scv.tl.recover_dynamics(adata, n_jobs = 6)
        scv.tl.velocity(adata, mode="dynamical")
        # scv.tl.velocity_graph(adata)
        # scv.pl.velocity_embedding_stream(adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4)
    else:
        scv.tl.velocity(adata)

    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis = "pca")
    scv.tl.velocity_embedding(adata, basis = "umap")

    cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
    cr.pl.terminal_states(adata, basis = "pca")

    cr.tl.initial_states(adata, cluster_key=None)
    cr.pl.initial_states(adata, basis = "pca", discrete=True)

    initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
    initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

    cr.tl.lineages(adata)
    cr.pl.lineages(adata, same_plot=False)
    lineage = adata.obsm['to_terminal_states']
    lineage_name = [x for x in lineage.names]
    lineage = np.array(lineage)
    max_lineage = np.argmax(lineage, axis = 1)
    max_lineage = np.array([x for x in max_lineage])

    if use_dynamic:
        scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
        scv.pl.scatter(adata,
            basis = "pca",
            color=["latent_time"],
            fontsize=16,
            cmap="viridis",
            perc=[2, 98],
            colorbar=True,
            rescale_color=[0, 1],
            title=["latent time"],
        )
        pt = adata.obs["latent_time"]
    else:
        
        root_idx = initial_idx[0]
        adata.uns["iroot"] = root_idx
        sc.tl.dpt(adata)
        scv.pl.scatter(adata,
            color=["dpt_pseudotime"],
            fontsize=16,
            cmap="viridis",
            perc=[2, 98],
            colorbar=True,
            rescale_color=[0, 1],
            title=["dpt pseudotime"],
        )

        pt = adata.obs["dpt_pseudotime"].values
    # pt_final = pd.DataFrame(data = np.nan, index = adata.obs.index, columns=lineage_name)
    # for i, x in enumerate(max_lineage):
    #     pt_final.iloc[i,x] = pt[i]

    pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
    pt_final.to_csv(path2 + data + "_latent_cellrank.tsv", sep = "\t", na_rep='NA')


# In[]
# real datasets
# hema
use_dynamic = True

adata = anndata.read_h5ad("../data/real/hema/adata_day4_clust.h5ad")

scv.tl.recover_dynamics(adata, n_jobs = 6)
scv.tl.velocity(adata, mode="dynamical")

# scv.tl.umap(adata)
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata, basis = "umap")

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, basis = "umap", discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        basis = "umap",
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
    pt = adata.obs["latent_time"]
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

    pt = adata.obs["dpt_pseudotime"].values


pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
pt_final.to_csv("../results/cellrank/real/hema/hema_latent_cellrank.tsv", sep = "\t", na_rep='NA')




# In[]
# dg

use_dynamic = True

# adata = anndata.read_h5ad("../data/real/DentateGyrus/10X43_1.h5ad")
# scv.tl.recover_dynamics(adata, n_jobs = 6)
# scv.tl.velocity(adata, mode="dynamical")

adata = anndata.read_h5ad("../data/real/DentateGyrus/dg_clust.h5ad")
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata, basis = "umap")

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, basis = "umap", discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        basis = "umap",
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
    pt = adata.obs["latent_time"]
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

    pt = adata.obs["dpt_pseudotime"].values


pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
pt_final.to_csv("../results/cellrank/real/dg/dg_latent_cellrank.tsv", sep = "\t", na_rep='NA')

# In[] pancreas

use_dynamic = True

adata = anndata.read_h5ad("../data/real/Pancreas/pe_clust.h5ad")
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata, basis = "umap")

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, basis = "umap", discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        basis = "umap",
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
    pt = adata.obs["latent_time"]
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

    pt = adata.obs["dpt_pseudotime"].values


pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
pt_final.to_csv("../results/cellrank/real/pe/pe_latent_cellrank.tsv", sep = "\t", na_rep='NA')


# In[] forebrain

use_dynamic = True

adata = anndata.read_h5ad("../data/real/forebrain/fb_clust.h5ad")
scv.tl.recover_dynamics(adata, n_jobs = 6)
scv.tl.velocity(adata, mode="dynamical")
gene_idx = ~np.isnan(np.sum(adata.layers["velocity"], axis = 0))
adata = adata[:,gene_idx]

scv.tl.umap(adata)
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis = "pca")
scv.tl.velocity_embedding(adata, basis = "umap")

cr.tl.terminal_states(adata, cluster_key=None, weight_connectivities=0.2)
cr.pl.terminal_states(adata, basis = "pca")

cr.tl.initial_states(adata, cluster_key=None)
cr.pl.initial_states(adata, basis = "pca", discrete=True)

initial_idx = [idx for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]
initial_val = [x for idx, x in enumerate(adata.obs["initial_states"].values) if x is not np.nan]

cr.tl.lineages(adata)
cr.pl.lineages(adata, same_plot=False)
lineage = adata.obsm['to_terminal_states']
lineage_name = [x for x in lineage.names]
lineage = np.array(lineage)
max_lineage = np.argmax(lineage, axis = 1)
max_lineage = np.array([x for x in max_lineage])

if use_dynamic:
    scv.tl.recover_latent_time(adata, root_key="initial_states_probs", end_key="terminal_states_probs")
    scv.pl.scatter(adata,
        basis = "pca",
        color=["latent_time"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["latent time"],
    )
    pt = adata.obs["latent_time"]
else:
    
    root_idx = initial_idx[0]
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    scv.pl.scatter(adata,
        color=["dpt_pseudotime"],
        fontsize=16,
        cmap="viridis",
        perc=[2, 98],
        colorbar=True,
        rescale_color=[0, 1],
        title=["dpt pseudotime"],
    )

    pt = adata.obs["dpt_pseudotime"].values


pt_final = pd.DataFrame(data = pt.values, index = adata.obs.index, columns = ["traj_0"])
pt_final.to_csv("../results/cellrank/real/forebrain/forebrain_latent_cellrank.tsv", sep = "\t", na_rep='NA')


# %%
