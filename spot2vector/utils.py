import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Build_Graph(adata, radius_cutoff=1.5, neighbors_cutoff=4,
                cutoff_type='radius', graph_type='spatial',
                symmetric=False, verbose=True):
    assert cutoff_type == 'radius' or cutoff_type == 'neighbors', print("please give a valid cutoff type.")
    assert graph_type == 'spatial' or graph_type == 'expression', print("please give a valid graph type.")

    if graph_type == 'spatial':
        if ('x_pixel' not in adata.obsm.keys()) or ('y_pixel' not in adata.obsm.keys()):
            adata.obsm["x_pixel"] = adata.obsm["spatial"][:, 0]
            adata.obsm["y_pixel"] = adata.obsm["spatial"][:, 1]
        else:
            pass
        x_pixel = adata.obsm["x_pixel"]
        y_pixel = adata.obsm["y_pixel"]
        used_data = np.column_stack((x_pixel, y_pixel))
    elif graph_type == 'expression':
        if 'X_pca' in adata.obsm:
            used_data = adata.obsm['X_pca']
        else:
            X = adata.X
            if isinstance(X, np.ndarray):
                pass
            else:
                X = X.toarray()
            used_data = X

    # unweighted adjacency matrix
    if cutoff_type == 'radius':
        A = np.sign(radius_neighbors_graph(used_data, radius_cutoff).toarray())
    elif cutoff_type == 'neighbors':
        A = np.sign(kneighbors_graph(used_data, neighbors_cutoff, metric="minkowski").toarray())

    np.fill_diagonal(A, 1)
    if symmetric:
        A = np.minimum(A, A.T)

    if isinstance(A, np.ndarray):
        pass
    else:
        A = A.toarray()
    adata.obsm[f'{graph_type}_graph'] = A

    if np.sum(A) == A.shape[0]:
        print(f"Get an empty graph, please consider changing the cutoff.")

    if verbose:
        print(
            f"The {graph_type} KNN grpah contains {A.shape[0]} nodes "
            f"and {np.sum(A) - A.shape[0]} edges.(except self-loop)")
        print(f"Average degree of {graph_type} graph: {np.sum(A) / A.shape[0] - 1:.2f}.")


def Graph_Stat_Plot(adata, ):
    if 'spatial_graph' in adata.obsm.keys():
        A = adata.obsm['spatial_graph']
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        _ = axes.hist((A.sum(1) - 1).flatten())
        plt.xlabel("node degree")
        plt.ylabel("node numbers")
        plt.title(f"Average degree of spatial KNN graph: {np.sum(A) / A.shape[0] - 1:.2f}.")
        plt.tight_layout()

    if 'expression_graph' in adata.obsm.keys():
        A = adata.obsm['expression_graph']
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        _ = axes.hist((A.sum(1) - 1).flatten())
        plt.xlabel("node degree")
        plt.ylabel("node numbers")
        plt.title(f"Average degree of expression KNN graph: {np.sum(A) / A.shape[0] - 1:.2f}.")
        plt.tight_layout()


def Clustering(adata, n_cluster=4, method='leiden', obsm_data='embedding', verbose=True):
    if verbose:
        print("Clustering...")
    embeddings = adata.obsm[obsm_data]
    adata_ = sc.AnnData(embeddings)
    sc.pp.neighbors(adata_, use_rep='X')
    resl = 0.01
    resr = 3
    res = (resr + resl) / 2
    step = 100

    if method == 'mclust':
        """\
        Clustering using the mclust algorithm.
        The parameters are the same as those in the R package mclust.
        """

        np.random.seed(12)
        import rpy2.robjects as robjects
        robjects.r.library("mclust")

        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(12)
        rmclust = robjects.r['Mclust']

        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[obsm_data]), n_cluster, 'EEE')
        mclust_res = np.array(res[-2])

        adata.obs[f"{obsm_data}_mclust"] = mclust_res
        adata.obs[f"{obsm_data}_mclust"] = adata.obs[f"{obsm_data}_mclust"].astype('int')
        adata.obs[f"{obsm_data}_mclust"] = adata.obs[f"{obsm_data}_mclust"].astype('category')

    if method == 'leiden':
        if verbose:
            print("Searching resolution...")
        sc.tl.leiden(adata_, resolution=res)
        while (len(np.unique(list(adata_.obs['leiden']))) != n_cluster) and step >= 0:
            if len(np.unique(list(adata_.obs['leiden']))) > n_cluster:
                resr = res
            else:
                resl = res
            res = (resr + resl) / 2

            sc.tl.leiden(adata_, resolution=res)
            step -= 1
            if verbose:
                print(f"searching resolution... current res = {res:.3f}")

        if len(np.unique(list(adata_.obs['leiden']))) != n_cluster:
            if verbose:
                print(f"searching failed, please reconstruct embeddings.")

        adata.obs[f"{obsm_data}_leiden"] = list(adata_.obs['leiden'])

    if method == 'louvain':
        if verbose:
            print("Searching resolution...")
        sc.tl.louvain(adata_, resolution=res)
        while (len(np.unique(list(adata_.obs['louvain']))) != n_cluster) and step >= 0:
            if len(np.unique(list(adata_.obs['louvain']))) > n_cluster:
                resr = res
            else:
                resl = res
            res = (resr + resl) / 2

            sc.tl.louvain(adata_, resolution=res)
            step -= 1
            if verbose:
                print(f"searching resolution... current res = {res:.3f}")

        if len(np.unique(list(adata_.obs['louvain']))) != n_cluster:
            if verbose:
                print(f"searching failed, please reconstruct embeddings.")

        adata.obs[f"{obsm_data}_louvain"] = list(adata_.obs['louvain'])


def Clustering_Metrics(adata, predict_obs='embedding', target_obs='domain_annotation', verbose=True):
    assert target_obs in adata.obs.keys(), print("domain_annotation is not available!")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd
    df = pd.DataFrame({
        'predict': adata.obs[predict_obs],
        'ground_truth': adata.obs[target_obs]
    })
    df = df.dropna()
    ari = adjusted_rand_score(df['ground_truth'], df['predict'], )
    nmi = normalized_mutual_info_score(df['ground_truth'], df['predict'], )
    adata.uns[f'{predict_obs}_ARI'] = ari
    adata.uns[f'{predict_obs}_NMI'] = nmi
    if verbose:
        print(f"ARI of {predict_obs} is: {ari}")
        print(f"NMI of {predict_obs} is: {nmi}")


def Matrix_Heatmap(matrix, fig_size=(5, 5), ax=None):
    if isinstance(matrix, torch.Tensor):
        vmax = torch.quantile(torch.abs(matrix), 0.9)
        # vmax = torch.max(torch.abs(matrix))
    else:
        # vmax = np.max(np.abs(matrix))
        vmax = np.quantile(np.abs(matrix).flatten(), 0.9)

    cmap = sns.diverging_palette(260, 10, as_cmap=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(matrix, cmap=cmap, vmax=vmax, vmin=-vmax, square=True, ax=ax)


def mix_embeddings(spatial_embeddings, expression_embeddings, lamda=0.5):
    if lamda == 0:
        embeddings = spatial_embeddings
    elif lamda == 1:
        embeddings = expression_embeddings
    else:
        embeddings = lamda * expression_embeddings + (1 - lamda) * spatial_embeddings

    return embeddings
