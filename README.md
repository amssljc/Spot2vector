# Spot2vector

Spot2vector is a novel computational framework that leverages a ZINB-based graph autoencoder for spatial clustering and data denoising. This method integrates both spatial and expression information to provide a comprehensive analysis of spatial transcriptomics (ST) data.

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

1. **Anaconda or Miniconda**: Ensure you have either Anaconda or Miniconda installed.
2. **CUDA version >= 11.8**: Required for GPU acceleration.
3. **NVIDIA GPU available**: Ensure you have a compatible NVIDIA GPU.


## Installation

For detailed installation instructions, please refer to [INSTALLATION.md](INSTALLATION.md)


## Quick Start

### 1. Data Preparation

The input data for Spot2vector should be an `AnnData` object, which can be loaded using `scanpy.read_h5ad`. The `AnnData` object must contain:

- **Preprocessed Expression Data**: The expression data should be preprocessed using standard single-cell RNA-seq preprocessing steps:
  ```python
  import scanpy as sc

  # Normalize total counts
  sc.pp.normalize_total(adata, target_sum=1e4)

  # Log transform the data
  sc.pp.log1p(adata)

  # Select highly variable genes
  sc.pp.highly_variable_genes(adata, n_top_genes=8000, flavor='seurat_v3')
  ```

- **Spatial coordinates**: The spatial coordinates should be stored in `adata.obsm["spatial"]`. The coordinates should be a 2D array of shape `(n_spots, 2)`.
- **Optional PCA**: For improved efficiency in constructing the expression similarity graph, you can perform PCA to obtain a low-dimensional representation: 
  ```python
  sc.pp.pca(adata, n_comps=10)
  ```

### 2. Graph Construction
Construct spatial and expression graphs using the following commands:
```python
import Spot2Vector

# Spatial graph based on spatial coordinates
Spot2Vector.Build_Graph(adata, radius_cutoff=150, cutoff_type='radius', graph_type='spatial')

# Expression graph based on expression similarity
Spot2Vector.Build_Graph(adata, neighbors_cutoff=4, cutoff_type='neighbors', graph_type='expression')
```
### 3. Model Training
Train the model using the following command:
```python
device = 'cuda:0'  # Specify the GPU device
Spot2Vector.Fit(adata, device=device)
```
### 4. Spatial Clustering (Spatial & Expression)
Perform spatial clustering using both the expression embeddings and spatial embeddings. The `n_clusters` parameter specifies the number of spatial domains, and users need to provide this value based on their dataset and biological knowledge.
```python
# Expression embeddings
Spot2Vector.Clustering(adata, obsm_data='exp_embeddings', method='mclust', n_cluster=n_clusters, verbose=False)

# Spatial embeddings
Spot2Vector.Clustering(adata, obsm_data='spa_embeddings', method='mclust', n_cluster=n_clusters, verbose=False)
```
### 5. Model Inference
Perform model inference to obtain the final embeddings:
```python
# lamda = 1 for expression, lamda = 0 for spatial
Spot2Vector.Infer(adata, lamda=0.2, device=device)
```
### 6. Spatial Clustering (Final Embeddings)
Perform the final spatial clustering using the combined embeddings:
```python
Spot2Vector.Clustering(adata, obsm_data='embeddings', method = 'mclust', n_cluster=n_clusters, verbose=False)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
