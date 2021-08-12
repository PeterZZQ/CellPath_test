rm(list = ls())
gc()

library(tidyverse)
library(dyngen)
library(anndata)
library(dyno)

setwd("~/Dropbox (GaTech)/Research/Projects/CellPath/additional_tests")

set.seed(7)

backbone <- backbone_binary_tree(
  num_modifications = 2
)

init <- initialise_model(
  backbone = backbone,
  num_cells = 2000,
  num_tfs = 100,
  num_targets = 0,
  num_hks = 0,
  simulation_params = simulation_default(census_interval = 10, ssa_algorithm = ssa_etl(tau = 300 / 3600)),
  verbose = FALSE
)
out <- generate_dataset(init, make_plots = TRUE)
dataset <- out$dataset
model <- out$model

plot_gold_mappings(model, do_facet = FALSE) + scale_colour_brewer(palette = "Dark2")

dataset <- as_dyno(model)
plot_dimred(dataset)

ad <- as_anndata(model)
# ad$write_h5ad("./additional_data/dyngen_tree/binary_tree3.h5ad")