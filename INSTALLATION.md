
## Installation Guidance

### 0. Requirements

1. Anaconda or Miniconda is installed. 
2. CUDA version >= 11.8. (To install **torch >= 2.0**)
3. NVIDIA GPU available.

### 1. Configure Conda Environment

1. Create conda environment 

    ```bash
    conda create -n spot2vector python=3.9
    ```

2. Activate conda environment 

    ```bash
    conda activate spot2vector
    ```

### 2. Configure R Environment

1. Install R and R-essentials:

    ```bash
    conda install r-base r-essentials
    ```

2. Start R console:

    ``` bash
    R
    ```
    Note: For someone runs R in a docker environment (under root), try to run R with below command:
    ```bash
    LC_ALL=C.UTF-8 R
    # instead of just `R`
    ```



3. Install `mclust` package:

    ```r
    install.packages("mclust")
    ```

4. Check if `mclust` is installed successfully:

    ```r
     library(mclust)
     ```

5. Quit R console:

    ```r
    q()
    ```

### 3. Configure Python Environment

1. Install torch, for example: 
    **Note: Match the torch version with CUDA!!!**
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    

2. Check if `torch` installed with cuda version successfully:
    ```
    python -c "import torch; print(torch.cuda.is_available())"
    ```

3. Install PyG:
    ```
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
    ```

4. Check if `torch_geometric` installed successfully:
    ```
    python -c "import torch_geometric; print('PyG is installed:', torch_geometric.__version__)"
    ```

5. Install `rpy2`:
    ```
    conda install rpy2
    ```

6. Install `scanpy`:
    ```
    pip install scanpy
    ```

7. Install skmisc to use 'seurat_v3' for gene selection.
    ```
    pip install --user scikit-misc
    ```
### 3. Setup Spot2Vector
