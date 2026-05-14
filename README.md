# PolyGraph

**PolyGraph** is a graph-based index for approximate nearest-neighbor search (ANNS) over **multi-vector data** under flexible query weights.

Each object contains multiple vector fields, and each query specifies a weight vector over these fields. PolyGraph selects representative weights, builds one edge group for each representative weight, prunes intra- and inter-group redundant edges, and performs weight-aware greedy search (WAGS) by activating only query-relevant edge groups.

PolyGraph is the official implementation of:

> **PolyGraph: An Efficient Multi-Vector Index for Approximate Nearest-Neighbor Search on Multi-Vector Data**  
> Mengtong Xu, James Pan, Guoliang Li

## Datasets

### Base and Query Data

The paper evaluates PolyGraph on four real-world million-scale multi-vector datasets. The original data sources are listed below. For convenience, we also provide the processed files, including base vectors, query vectors, and path-info files. All base and query vectors are stored in `fvecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for details about `fvecs`.

| Dataset name `<DATASET>` | Paper dataset   | Raw Data Source | Number of fields | Dimensions | Path-info file | Download (base and query) |
| ------------------------ | ------------- | ------ | ---------------: | ---------- | -------------- | ------------------------- |
| `ImageText` | Image-Text | [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) | 2 | `(768, 768)` | `dataset/path_info_CC1MNorm_2field_100W.txt` |[ImageText_data.tar.gz](https://cloud.tsinghua.edu.cn/f/ac05841eb31046d69962/?dl=1) |
| `QA2` | Question-Answer | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4 | `(384, 512, 384, 512)` | `dataset/path_info_LMSYSNorm_4field_100W.txt` | [QA2_data.tar.gz](https://cloud.tsinghua.edu.cn/f/b4559a7b7b9840d5864f/?dl=1) |
| `Wiki` | Wikipedia | [Wikipedia Structured Contents](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data) | 6 |  `(384, 384, 384, 384, 384, 384)` |`dataset/path_info_EnwikiNorm_6field_100W.txt` | [Wiki_data.tar.gz](https://cloud.tsinghua.edu.cn/f/e27c0615f29641e080f5/?dl=1)  |
| `Protein` | Protein | [UniProt](https://www.uniprot.org/) | 8 | `(400, 128, 128, 128, 128, 128, 128, 320)` | `dataset/path_info_ProteinNorm_8field_100W.txt` | [Protein_data.tar.gz](https://cloud.tsinghua.edu.cn/f/82fa7bb77764436ca945/?dl=1)  |

After downloading the processed files, place each dataset folder under `dataset/VectorFiles/`, or update the corresponding path-info file with your local file paths. PolyGraph uses the path-info file to locate the base and query vector files of each field.

Each path-info file contains one block per vector field. In the $i$-th block, the first and second lines specify the paths to the base vectors and query vectors of the $i$-th field, respectively. A two-field example is shown below:

```text
/path/to/field_1_base.fvecs
/path/to/field_1_query.fvecs

/path/to/field_2_base.fvecs
/path/to/field_2_query.fvecs
```

### Ground-Truth Files

All ground-truth files are stored in `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for details about `ivecs`.

In the paper, PolyGraph uses a default query workload that contains all `2^m - 1` non-empty field-participation patterns for a dataset with `m` fields. In this implementation, active fields are assigned equal positive weights, which is rank-equivalent to the normalized workload $`\mathcal{W}_{\mathrm{default}}`$ used in the paper. The ground-truth file for the $i$-th weight configuration in the default workload should be stored as:

```text
dataset/Ground-truth/<DATASET>/<i>-output.ivecs
```

If the corresponding ground-truth file is not available, PolyGraph computes the exact results by brute force.

Download links for the ground-truth files of the default query workload are listed below:

| Dataset name `<DATASET>` | Paper dataset | Number of ground-truth files | Download (ground-truth) |
| ------------------------ | ------------- | ----------------------------: | ----------------------- |
| `ImageText`              | Image-Text      | 3 | [ImageText_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/d7f46fba69b448278a37/?dl=1) |
| `QA2`                    | Question-Answer | 15 | [QA2_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/e6c0f3faf5a74877a5a0/?dl=1) |
| `Wiki`                   | Wikipedia       | 63 | [Wiki_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/6b8baba61f714c6faaf4/?dl=1)  |
| `Protein`                | Protein         | 255 | [Protein_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/15c2c326aa974ce9bdf3/?dl=1)  |




## Main Parameters

| Parameter                 | In-paper notation | Default / typical value | Description                                         |
| ------------------------- | ----------------- | ----------------------: | --------------------------------------------------- |
| `-total_sim_thresh`       | $\tau$            |                  `0.95` | Representative-weight coverage threshold.           |
| `-rela_sim_thresh`        | $c$               |                   `0.5` | Related-group threshold used during construction.   |
| `-search_rela_sim_thresh` | $c$               |                   `0.5` | Edge-group activation threshold used during search. |
| `-R_refine`               | $R$               |       dataset-dependent | Construction out-degree budget.                     |
| `-L_refine`               | $L$               |       dataset-dependent | Construction candidate-list budget.                 |
| `k`                       | $k$               |                    `20` | Number of nearest neighbors for recall evaluation.  |

## Usage

### Requirements

- CMake >= 3.10
- C++14-compatible compiler
- OpenMP
- Boost >= 1.55
- Python 3

### Compile on Linux

Clone the repository, create a Python virtual environment, install the Python dependencies, and compile PolyGraph with:

```bash
# Clone the repository
git clone https://github.com/TsinghuaDatabaseGroup/PolyGraph.git
cd PolyGraph

# Create a Python virtual environment and install the Python dependencies
python3 -m venv pythonEnv_ForPG
source pythonEnv_ForPG/bin/activate
pip install -r include/python_file/requirements.txt

# Create output directories
mkdir -p build myIndex include/python_file/nohup_logs include/python_file/backup_clusterGroups

# Compile
cd build
cmake ..
make -j
```

### Build a PolyGraph index

Run the following command from the `build/` directory to construct a PolyGraph index with the default parameter values ($\tau=0.95$ and $c=0.5$):

```bash
./main PolyGraph <DATASET> build
```

To specify different values for `tau` and `c`, use:

```bash
./main PolyGraph <DATASET> build -total_sim_thresh <tau> -rela_sim_thresh <c>
```

### Evaluate PolyGraph at Recall@20 with WAGS

After building the index, run the following command from the `build/` directory to evaluate PolyGraph with WAGS under the default query workload. By default, WAGS uses the activation threshold `c = 0.5`. Note that to use the default query workload, set the first line of the corresponding `dataset/useWeight/useWeightEachQuery_[...].txt` file to `0` if the file exists. The command evaluates all non-empty field-participation patterns and reports the recall-latency performance following the experimental setting in the paper. The output logs include query latency, Recall@20, average query path length, candidate-set statistics, and the average number of distance evaluations.

```bash
./main PolyGraph <DATASET> all_recall_search 20
```

To specify a different WAGS activation threshold `c`, use:

```bash
./main PolyGraph <DATASET> all_recall_search 20 -search_rela_sim_thresh <c>
```


### Built-in Baselines

The repository also includes several built-in baselines used in the paper.

```bash
# Build an index
./main <INDEX_NAME> <DATASET> build

# Evaluate Recall@k
./main <INDEX_NAME> <DATASET> all_recall_search <k>
```

| Command name `<INDEX_NAME>`  | Paper name      |
| ---------------------------- | --------------- |
| `hnsw_fusion`                | `HNSW_Fusion`   |
| `vamana_fusion`              | `Vamana_Fusion` |
| `vamana_equNoTotal`          | `Vamana_Union`  |
| `vamana_allWeight`           | `Vamana_Poly`   |
| `vamana_oracle`              | `Oracle_Vamana` |

DEG and HJG are external baselines and should be run using their official repositories.