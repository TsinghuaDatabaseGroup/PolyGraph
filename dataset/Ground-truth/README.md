# Ground Truths

PolyGraph uses a default query workload that contains all `2^m - 1` non-empty field-participation patterns for a dataset with `m` fields. In the implementation, active fields are assigned equal positive weights, which is rank-equivalent to the normalized workload $`\mathcal{W}_{\mathrm{default}}`$ used in the paper. The ground-truth file for the $i$-th weight configuration in $`\mathcal{W}_{\mathrm{default}}`$ should be stored as:

```text
dataset/Ground-truth/<DATASET>/<i>-output.ivecs
```

If the corresponding ground-truth file is not available, PolyGraph computes the exact results by brute force.

Download links for the ground-truth files of the default query workload are listed below:
| Dataset name `<DATASET>` | Paper dataset | Number of ground-truth files | Download (ground-truth) |
| ------------------------ | --------------- | --------------- | -------------- |
| `ImageText`              | Image-Text      | 3 | [ImageText_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/d7f46fba69b448278a37/?dl=1) |
| `QA2`                    | Question-Answer | 15 | [QA2_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/e6c0f3faf5a74877a5a0/?dl=1) |
| `Wiki`                   | Wikipedia       | 63 | [Wiki_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/6b8baba61f714c6faaf4/?dl=1)  |
| `Protein`                | Protein         | 255 | [Protein_GT.tar.gz](https://cloud.tsinghua.edu.cn/f/15c2c326aa974ce9bdf3/?dl=1)  |
