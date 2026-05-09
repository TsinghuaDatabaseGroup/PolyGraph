# Datasets

The original data sources used in the paper are listed below. For convenience, we also provide the processed dataset files, including base vectors, query vectors, and path-info files. Note that, all base and query vectors are stored in `fvecs` format. Please refer to the [YAEL file-format description](http://yael.gforge.inria.fr/file_format.html) for details.

| Dataset name `<DATASET>` | Paper dataset   | Source | Number of fields | Path-info file | Download (base and query) |
| ------------------------ | --------------- | ------ | --------------- | -------------- | ------------------------------ |
| `ImageText` | Image-Text | [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) | 2 | `./path_info_CC1MNorm_2field_100W.txt` |[ImageText_data.tar.gz](https://cloud.tsinghua.edu.cn/f/ac05841eb31046d69962/?dl=1) |
| `QA2` | Question-Answer | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 4 | `./path_info_LMSYSNorm_4field_100W.txt` | [QA2_data.tar.gz](https://cloud.tsinghua.edu.cn/f/b4559a7b7b9840d5864f/?dl=1) |
| `Wiki` | Wikipedia | [Wikipedia Structured Contents](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data) | 6 | `./path_info_EnwikiNorm_6field_100W.txt` | [Wiki_data.tar.gz](https://cloud.tsinghua.edu.cn/f/e27c0615f29641e080f5/?dl=1)  |
| `Protein` | Protein | [UniProt](https://www.uniprot.org/) | 8 | `./path_info_ProteinNorm_8field_100W.txt` | [Protein_data.tar.gz](https://cloud.tsinghua.edu.cn/f/82fa7bb77764436ca945/?dl=1)  |

PolyGraph reads each multi-vector dataset through a path-info file. Each path-info file contains one block per field. In the $i$-th block, the first and second lines specify the paths to the base vectors and query vectors of the $i$-th field, respectively. A two-field example is shown below:

```text
/path/to/field_1_base.fvecs
/path/to/field_1_query.fvecs

/path/to/field_2_base.fvecs
/path/to/field_2_query.fvecs
```
