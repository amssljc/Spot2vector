# Spot2vector method for ST data analysis

Spot2vector is a novel computational framework that leverages a ZINB-based graph autoencoder for spatial clustering and data denoising.


## Authors

- amssljc@163.com
- yujiating@amss.ac.cn


## Pipeline

<p align="center">
  <a href="https://github.com/amssljc/Spot2vector/">
    <img src="image/Spot2vector.svg" alt="Logo">
  </a>
</p>


## Requirements

1. Anaconda or Miniconda. 
2. CUDA version >= 11.8.
3. NVIDIA GPU available.


## Installation

Installation tutorial is in [INSTALLATION.md](INSTALLATION.md)


## Quick Start
1. Graph construction
```
import Spot2Vector

Spot2Vector.Build_Graph(adata, radius_cutoff=150, cutoff_type='radius', graph_type='spatial')
Spot2Vector.Build_Graph(adata, neighbors_cutoff=4, cutoff_type='neighbors', graph_type='expression')
```
2. Model training
```
device = 'cuda:0'
Spot2Vector.Fit(adata, device=device)
```
3. Spatial clustering (spatial & expression)
```
Spot2Vector.Clustering(adata, obsm_data='exp_embeddings', method='mclust', n_cluster=n_clusters, verbose=False)
Spot2Vector.Clustering(adata, obsm_data='spa_embeddings', method='mclust', n_cluster=n_clusters, verbose=False)
```
4. Model inference
```
# lamda = 1 for expression, lamda = 0 for spatial
Spot2Vector.Infer(adata, lamda=0.2, device=device)
```
5. Spatial clustering (final embeddings)
```
Spot2Vector.Clustering(adata, obsm_data='embeddings', method = 'mclust', n_cluster=n_clusters, verbose=False)
```