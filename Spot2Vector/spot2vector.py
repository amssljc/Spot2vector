import torch.nn as nn
import torch.nn.functional as F
from .models import LinearCoder, GraphAttentionEncoder, GraphAttentionDecoder
from .lossfunctions import zinb_loss
import torch
import random
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data
import os
from .utils import mix_embeddings, set_seed
import numpy as np


class AE_st(nn.Module):
    def __init__(self,
                 gene_dim,
                 embedding_dim=32,
                 hidden_dims_E_st=[512],
                 hidden_dims_D_st=[256, 512],
                 lamda=0.5,
                 dropout=0.0,
                 heads=1):
        super().__init__()
        self.gene_dim = gene_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims_E_st = hidden_dims_E_st
        self.hidden_dims_D_st = hidden_dims_D_st
        self.lamda = lamda
        self.heads = heads
        self.dropout = dropout

        self.GE1 = GraphAttentionEncoder(
            layer_dims=[self.gene_dim] + self.hidden_dims_E_st + [self.embedding_dim],
            dropout=self.dropout, heads=self.heads
        )
        self.GE2 = GraphAttentionEncoder(
            layer_dims=[self.gene_dim] + self.hidden_dims_E_st + [self.embedding_dim],
            dropout=self.dropout, heads=self.heads
        )

        self.Z2pi = LinearCoder(layer_dims=[self.embedding_dim] + self.hidden_dims_D_st + [self.gene_dim])
        self.Z2mu = LinearCoder(layer_dims=[self.embedding_dim] + self.hidden_dims_D_st + [self.gene_dim])
        self.Z2theta = LinearCoder(layer_dims=[self.embedding_dim] + self.hidden_dims_D_st + [self.gene_dim])

    def forward(self, x, expression_edge_index, spatial_edge_index):
        spatial_embeddings = self.GE1(x, spatial_edge_index)
        expression_embeddings = self.GE2(x, expression_edge_index)
        embeddings = mix_embeddings(spatial_embeddings, expression_embeddings, self.lamda)

        pi = self.Z2pi(embeddings)
        pi = F.sigmoid(pi)
        mu = self.Z2mu(embeddings)
        mu = F.softplus(mu)
        theta = self.Z2theta(embeddings)
        theta = torch.exp(theta)

        return embeddings, expression_embeddings, spatial_embeddings, pi, mu, theta


def Fit(adatast,
        embedding_dim=32,
        hidden_dims_E_st=[512],
        hidden_dims_D_st=[256, 512],
        reg_theta=1e-2,
        device='cuda', lamda=0.5,
        lr_st=1e-4, weight_decay=1e-4,
        dropout=0.0, heads=1,
        max_epochs_st=1500,
        verbose=False, save_history=True, seed=1,
        ):
    if seed:
        set_seed(seed)

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        from torch.backends import cudnn
        import numpy as np
        # cudnn.benchmark = False            # if benchmark=True, deterministic will be False
        # cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    X_st = adatast.X
    if isinstance(X_st, np.ndarray):
        pass
    else:
        X_st = X_st.toarray()

    assert 'spatial_graph' in adatast.obsm.keys(), print(
        "there is no spatial graph, please call Build_Graph first.")
    assert 'expression_graph' in adatast.obsm.keys(), print(
        "there is no expression graph, please call Build_Graph first.")

    A1 = adatast.obsm['expression_graph']
    A2 = adatast.obsm['spatial_graph']

    A1 = torch.as_tensor(A1)
    A2 = torch.as_tensor(A2)

    traindata_expression = Data(torch.FloatTensor(X_st), A1.nonzero().t().contiguous().long()).to(device)
    traindata_spatial = Data(torch.FloatTensor(X_st), A2.nonzero().t().contiguous().long()).to(device)
    X_st = torch.as_tensor(X_st).to(device)

    model_st = AE_st(
        gene_dim=adatast.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims_E_st=hidden_dims_E_st,
        hidden_dims_D_st=hidden_dims_D_st,
        lamda=lamda,
        dropout=dropout,
        heads=heads,
    )
    model_st.train()
    model_st = model_st.to(device)

    optimizer2 = torch.optim.Adam(model_st.parameters(), lr=lr_st, weight_decay=weight_decay)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=0.9, patience=50, min_lr=1e-5
    )

    training_history_df = pd.DataFrame(columns=['epoch', 'loss_ZINB', 'loss_total'])
    if verbose:
        print("Training Spot2Vector...")
    pbar = tqdm(range(max_epochs_st), total=max_epochs_st, desc='Training', unit='epoch')
    loss_f = zinb_loss

    for epoch in pbar:
        optimizer2.zero_grad()
        embeddings_st, expression_embeddings_st, spatial_embeddings_st, pi_st, mu_st, theta_st = model_st(
            traindata_expression.x, traindata_expression.edge_index, traindata_spatial.edge_index)
        loss_ZINB = loss_f(X_st, mu_st, pi_st, theta_st, reg_theta)

        loss_total = loss_ZINB
        loss_total.backward()
        optimizer2.step()
        pbar.set_postfix(epoch=f"{epoch}", loss=f"{loss_total.detach():.3f}",
                         lr=f"{optimizer2.param_groups[0]['lr']:.4f}")

        scheduler2.step(loss_total)
        if optimizer2.param_groups[0]['lr'] < 1e-5:
            break

        if save_history:
            training_history_df = pd.concat([training_history_df,
                                             pd.DataFrame({
                                                 'epoch': [epoch],
                                                 'loss_ZINB': [float(loss_ZINB.detach().cpu().numpy())],
                                                 'loss_total': [float(loss_total.detach().cpu().numpy())],
                                             })],
                                            ignore_index=True)

    if save_history:
        adatast.uns['training_history_df_st'] = training_history_df

    model_st.eval()

    embeddings_st, expression_embeddings_st, spatial_embeddings_st, pi_st, mu_st, theta_st = model_st(
        traindata_expression.x, traindata_expression.edge_index, traindata_spatial.edge_index)

    adatast.obsm['exp_embeddings'] = expression_embeddings_st.detach().cpu().numpy()
    adatast.obsm['spa_embeddings'] = spatial_embeddings_st.detach().cpu().numpy()

    adatast.model_st = model_st


def Infer(adatast,
          lamda=0.5,
          device='cuda'
          ):
    import numpy as np

    # A
    A1 = adatast.obsm['expression_graph']
    A2 = adatast.obsm['spatial_graph']
    A1 = torch.as_tensor(A1)
    A2 = torch.as_tensor(A2)

    # X_st
    X_st = adatast.X
    if isinstance(X_st, np.ndarray):
        pass
    else:
        X_st = X_st.toarray()

    # model_st
    model_st = adatast.model_st

    # testdata
    testdata_expression = Data(torch.FloatTensor(X_st), A1.nonzero().t().contiguous().long()).to(device)
    testdata_spatial = Data(torch.FloatTensor(X_st), A2.nonzero().t().contiguous().long()).to(device)

    model_st.lamda = lamda

    model_st.eval()
    embeddings_st, expression_embeddings_st, spatial_embeddings_st, pi_st, mu_st, theta_st = model_st(
        testdata_expression.x, testdata_expression.edge_index, testdata_spatial.edge_index)

    adatast.obsm['embeddings'] = embeddings_st.detach().cpu().numpy()
    adatast.obsm['pi'] = pi_st.detach().cpu().numpy()
    adatast.obsm['mu'] = mu_st.detach().cpu().numpy()
    adatast.obsm['theta'] = theta_st.detach().cpu().numpy()
