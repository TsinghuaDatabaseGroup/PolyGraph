# PolyGraph

**PolyGraph** is a graph-based index for approximate nearest-neighbor search (ANNS) over **multi-vector data** under flexible query weights.

Each object contains multiple vector fields, and each query specifies a weight vector over these fields. PolyGraph selects representative weights, builds one edge group for each representative weight, prunes intra- and inter-group redundant edges, and performs weight-aware greedy search (WAGS) by activating only query-relevant edge groups.

PolyGraph is the official implementation of:

> **PolyGraph: An Efficient Multi-Vector Index for Approximate Nearest-Neighbor Search on Multi-Vector Data**  
> Mengtong Xu, James Pan, Guoliang Li

## Datasets

The paper evaluates PolyGraph on four real-world million-scale multi-vector datasets. We provide processed base vectors, query vectors, path-info files, and ground-truth files separately because the generated embedding files are large.

Please refer to:

- [`dataset/VectorFiles/README.md`](dataset/VectorFiles/README.md) for dataset sources, downloads, and path-info files.
- [`dataset/Ground-truth/README.md`](dataset/Ground-truth/README.md) for ground-truth files and the default query workload.

| Dataset name `<DATASET>` | Paper dataset | Number of fields | Dimensions |
| ------------------------ | ------------- | ---------------: | ---------- |
| `ImageText` | Image-Text | 2 | `(768, 768)` |
| `QA2` | Question-Answer | 4 | `(384, 512, 384, 512)` |
| `Wiki` | Wikipedia | 6 | `(384, 384, 384, 384, 384, 384)` |
| `Protein` | Protein | 8 | `(400, 128, 128, 128, 128, 128, 128, 320)` |

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

### Build a PolyGraph index with $\tau=0.95$ and $c=0.5$

Then, you can run the following instructions for build graph index.

```bash
cd PolyGraph/build
./main PolyGraph <DATASET> build -total_sim_thresh 0.95 -rela_sim_thresh 0.5
```

### Evaluate PolyGraph at Recall@20 with WAGS ($c=0.5$)

After building the index, run the following command to evaluate PolyGraph under the default query workload. If a corresponding `useWeight/useWeightEachQuery_[...].txt` file exists, set its first line to `0` to enable the default workload. The `all_recall_search` command then evaluates all non-empty field-participation patterns and reports the recall-latency performance following the experimental setting in the paper. The output logs include query latency, Recall@20, average query path length, and the average number of distance evaluations.

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