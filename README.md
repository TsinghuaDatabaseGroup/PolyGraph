# PolyGraph

**PolyGraph** is a graph-based index for approximate nearest-neighbor search (ANNS) over **multi-vector data** under flexible query weights.

Each object contains multiple vector fields, and each query specifies a weight vector over these fields. PolyGraph selects representative weights, builds one edge group for each representative weight, prunes intra- and inter-group redundant edges, and performs weight-aware greedy search (WAGS) by activating only query-relevant edge groups.

PolyGraph is the official implementation of:

> **PolyGraph: An Efficient Multi-Vector Index for Approximate Nearest-Neighbor Search on Multi-Vector Data**  
> Mengtong Xu, James Pan, Guoliang Li

## Datasets

Each dataset can be derived from the following public data sources.

| Dataset name `<DATASET>` | Paper dataset   | Source | Number of fields | Path-info file |
| ------------------------ | --------------- | ------ | ---------------: | -------------- |
| `ImageText` | Image-Text | [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) | 2 | `dataset/path_info_CC1MNorm_2field_100W.txt` |
| `QA2` | Question-Answer | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4 | `dataset/path_info_LMSYSNorm_4field_100W.txt` |
| `Wiki` | Wikipedia | [Wikipedia Structured Contents](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data) | 6 | `dataset/path_info_EnwikiNorm_6field_100W.txt` |
| `Protein` | Protein | [UniProt](https://www.uniprot.org/) | 8 | `dataset/path_info_ProteinNorm_8field_100W.txt` |

PolyGraph reads each multi-vector dataset through a path-info file. Each path-info file contains one block per field. In the $i$-th block, the first and second lines specify the paths to the base vectors and query vectors of the $i$-th field, respectively. A two-field example is shown below:

```text
/path/to/field_1_base.fvecs
/path/to/field_1_query.fvecs

/path/to/field_2_base.fvecs
/path/to/field_2_query.fvecs
```

Base and query vectors should be stored in `fvecs` format, and ground-truth files should be stored in `ivecs` format. Please refer to the [YAEL file-format description](http://yael.gforge.inria.fr/file_format.html) for details about `fvecs` and `ivecs`.

PolyGraph uses a default query workload that contains all `2^m - 1` non-empty field-participation patterns for a dataset with `m` fields. In the implementation, active fields are assigned equal positive weights, which is rank-equivalent to the normalized workload $`\mathcal{W}_{\mathrm{default}}`$ used in the paper. The ground-truth file for the $i$-th weight configuration in $`\mathcal{W}_{\mathrm{default}}`$ should be stored as:

```text
dataset/Ground-truth/<DATASET>/<i>-output.ivecs
```

If the corresponding ground-truth file is not available, PolyGraph computes the exact results by brute force.

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

Create a Python virtual environment and install the Python dependencies with:

```bash
python3 -m venv pythonEnv_ForSI
source pythonEnv_ForSI/bin/activate
pip install -r include/python_file/requirements.txt
```

### Compile on Linux

```bash
git clone https://github.com/TsinghuaDatabaseGroup/PolyGraph.git
cd PolyGraph
mkdir -p build myIndex include/python_file/nohup_logs include/python_file/backup_clusterGroups
cd build
cmake ..
make -j
```

The executable is generated as `build/main`.

### Quick Start of PolyGraph

Run the following commands from the `build/` directory.

#### Build a PolyGraph index with $\tau=0.95$ and $c=0.5$

```bash
cd PolyGraph/build
./main PolyGraph <DATASET> build -total_sim_thresh 0.95 -rela_sim_thresh 0.5
```

#### Evaluate PolyGraph at Recall@20 with WAGS ($c=0.5$)

```bash
cd PolyGraph/build
./main PolyGraph <DATASET> all_recall_search 20 -search_rela_sim_thresh 0.5
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

## Citation

If you use PolyGraph in your research, please cite:

```bibtex
@article{xu2026polygraph,
  title   = {PolyGraph: An Efficient Multi-Vector Index for Approximate Nearest-Neighbor Search on Multi-Vector Data},
  author  = {Xu, Mengtong and Pan, James and Li, Guoliang},
  journal = {Proceedings of the VLDB Endowment},
  year    = {2026}
}
```