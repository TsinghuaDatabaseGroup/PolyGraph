#  ------------------------
# | 这个版本只进行representative weight selection，correlation decision；构建顺序根据representative weight selection选出来的顺序进行。
#  ------------------------
#  ------------------------
# | import
#  ------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import importlib.util
import sys
from datetime import datetime  # ✅ 用于生成“当前时间”字符串
import numpy as np
import argparse
import re
from joblib import Parallel, delayed




# ===== 1.1) 加载自定义函数 =====
# 下方操作的目的为： from /home/mengtong/Link09/LargeIndex/simple_exp/50817_cluster_GMM/AnalysePathInfo.py import load_base_paths as loadBathPath
ANALYSE_PATH = Path("../include/python_file/AnalysePathInfo.py")
# ANALYSE_PATH = Path("./AnalysePathInfo.py")
spec = importlib.util.spec_from_file_location("AnalysePathInfo", str(ANALYSE_PATH))
AnalysePathInfo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AnalysePathInfo)
loadBathPath = AnalysePathInfo.load_base_paths

# ===== 1.2) 动态加载 fun 模块（文件名含 '-' 无法常规 import，用 importlib 路径加载） =====
FUN_PATH = Path("../include/python_file/ClusterGMM-0817-fun.py")
# FUN_PATH = Path("./ClusterGMM-0817-fun.py")
assert FUN_PATH.exists(), f"Module file not found: {FUN_PATH}"
_spec = importlib.util.spec_from_file_location("fun_0817", str(FUN_PATH))
fun_0817 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fun_0817)






#  ------------------------
# | 超参数
#  ------------------------
cluster_group_txt = "../include/python_file/cluster_groups.txt"

def _parse_args():
    # -- 默认值 ---------------------------------------
    # sample_size=10000            # 抽样样本数
    # num_query=300                # 每个权重取的查询个数
    # n_jobs=-1                    # 并行核数
    # random_state=42              # 随机种子
    # rela_use_intersect = true         (true = --rela_use_intersect; false = --no-rela_use_intersect)
    # total_sim_thresh = 0.95
    # rela_sim_thresh = 0.5
    # max_group = 0                # 最多可以选择出几个组
    # ------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--total_sim_thresh", type=float, default=0.95, help="阈值，例如 0.95")
    p.add_argument("--rela_sim_thresh", type=float, default=0.5, help="阈值，例如 0.5")
    p.add_argument("--max_group", type=int, default=0)

    # 两个互斥 flag：默认 True，可用 --no-rela-use-intersect 关闭
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rela_use_intersect", dest="rela_use_intersect", action="store_true",  help="使用交集关系（默认）")
    g.add_argument("--no-rela_use_intersect", dest="rela_use_intersect", action="store_false", help="不用交集关系，走阈值")
    p.set_defaults(rela_use_intersect=True)

    p.add_argument("--sample_size", type=int, default=10000)
    p.add_argument("--num_query",   type=int, default=300)
    p.add_argument("--n_jobs",      type=int, default=-1)
    p.add_argument("--random_state",type=int, default=42)

    # ---- 2026.01.15: Query-weight input & GS-corr cache ----
    p.add_argument("--use_weight_each_query_txt", type=str,
                   default="../dataset/useWeight/useWeightEachQuery.txt",
                   help="优先从该文件读取 query weight vectors；失败则 fallback 到 2^m-1 个 0/1 weight")
    p.add_argument("--search_corr_thresh", type=float, default=0.3,
                   help="构建阶段缓存的 GS-corr 阈值 tau；查询阶段若 tau 相同可直接用缓存的 enabled groups")
    p.add_argument("--dedup_decimals", type=int, default=6,
                   help="去重/建立 key 时对 query weight 的 round 精度（小数位）")
    p.add_argument("--out_dir", type=str,
                   default="../include/python_file/backup_clusterGroups/",
                   help="cluster_groups 与 cache txt 的输出目录")

    # -- txt --
    p.add_argument("--txt_path", type=str,
                   default="../dataset/path_info.txt",
                   help="包括 basedata 和 querydata 路径的txt文件路径")

    p.add_argument("--forced_order_way", type=int,default=-1,
                    help=(
                        "Force representative order for 2-field case reps [1,0],[0,1],[1,1]. "
                        "Use 1..6 for the 6 permutations; -1 means keep original S_k order.\n"
                        "1:[1,0]->[0,1]->[1,1]\n"
                        "2:[1,0]->[1,1]->[0,1]\n"
                        "3:[0,1]->[1,0]->[1,1]\n"
                        "4:[0,1]->[1,1]->[1,0]\n"
                        "5:[1,1]->[1,0]->[0,1]\n"
                        "6:[1,1]->[0,1]->[1,0]"
                        )
                    )

    return p.parse_args()

_args = _parse_args()
sample_size=_args.sample_size            # 抽样样本数
num_query=_args.num_query                # 每个权重取的查询个数
n_jobs=_args.n_jobs                    # 并行核数
random_state=_args.random_state             # 随机种子

rela_use_intersect = _args.rela_use_intersect
total_sim_thresh = _args.total_sim_thresh
rela_sim_thresh = _args.rela_sim_thresh
max_group = _args.max_group

forced_order_way = _args.forced_order_way # 强制调整顺序

# ---- 2026.01.15: extra inputs for query weights & cache ----
use_weight_each_query_txt = _args.use_weight_each_query_txt
search_corr_thresh = _args.search_corr_thresh
# search_corr_thresh = rela_use_intersect # 2026.01.24, 就让search_corr_thresh = rela_use_intersect
dedup_decimals = _args.dedup_decimals
out_dir = _args.out_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)

# -- txt --
path_info_txt =  _args.txt_path


# （可选）做下范围校验
if not (0.0 <= total_sim_thresh <= 1.0):
    raise ValueError(f"--total_sim_thresh 应该在 [0,1]，收到 {total_sim_thresh}")






#  ------------------------
# | 读取数据
#  ------------------------
try:
    base_data_list = loadBathPath(path_info_txt)
except Exception as e:
    print(f"[WARN] 加载 Synthetic_100W 失败❗️❗️❗️，错误：{e}，尝试加载备用 path_info.txt")
    # base_data_list = loadBathPath("/home/mengtong/Link09/LargeIndex/dataset/NUS_WIDE_OBJ/path_info.txt")
    base_data_list = loadBathPath("../dataset/path_info.txt")
print(base_data_list)

# --- 0. 生成“当前时间”字符串用于输出文件命名（如 20250819-223045） ---
当前时间 = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"当前时间: {当前时间}")


#  ------------------------
# | 2026.01.15: 读取 query weight（优先 useWeightEachQuery.txt）+ 去重 + 缓存
#  ------------------------
def _fallback_all_binary_weights(num_fields: int):
    qlist = []
    for w in range(1, (1 << num_fields)):
        qlist.append([(1.0 if (w & (1 << j)) else 0.0) for j in range(num_fields)])
    return qlist

def load_query_weights_or_fallback(num_fields: int, path: str, verbose=True):
    """
    useWeightEachQuery.txt 格式：
      第一行：n（query weight vectors 数量）
      后续每行：一个 vector，空格分隔 float
    规则：
      - 若文件不存在 / 第一行是0 / 维度!=num_fields / 实际读取数!=n -> fallback 到 2^m-1 个 0/1 weight
    返回：(query_weight_list, source_str)
    """
    p = Path(path)
    if not p.exists():
        if verbose: print(f"[useWeight] file not found: {path}, fallback to all 0/1 weights.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() != ""]
    if len(lines) == 0:
        if verbose: print(f"[useWeight] empty file: {path}, fallback.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"

    try:
        n = int(lines[0].split()[0])
    except Exception:
        if verbose: print(f"[useWeight] invalid first line: {lines[0]!r}, fallback.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"

    if n <= 0:
        if verbose: print(f"[useWeight] first line is {n}, fallback.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"

    vecs = []
    for ln in lines[1:]:
        parts = ln.split()
        try:
            v = [float(x) for x in parts]
        except Exception:
            vecs = []
            break
        vecs.append(v)
        if len(vecs) >= n:
            break

    if len(vecs) != n:
        if verbose: print(f"[useWeight] expected n={n} but read {len(vecs)}, fallback.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"
    if any(len(v) != num_fields for v in vecs):
        if verbose:
            bad = [i for i,v in enumerate(vecs) if len(v) != num_fields][:5]
            print(f"[useWeight] dimension mismatch at lines {bad}, fallback.")
        return _fallback_all_binary_weights(num_fields), "fallback_binary"

    if verbose: print(f"[useWeight] loaded {n} query weights from {path}")
    return vecs, "useWeightEachQuery"

def dedup_query_weights(query_weight_list, decimals=6):
    """
    返回：(unique_list, first_idx, key_strs)
      - unique_list: 去重后的向量列表
      - first_idx  : 每个 unique 向量在原 list 的首次出现位置（用于从 sim_matrix 取 corr）
      - key_strs   : 每个 unique 向量的 key（稳定字符串）
    """
    seen = {}
    unique_list, first_idx, key_strs = [], [], []
    for idx, w in enumerate(query_weight_list):
        key = tuple(round(float(x), decimals) for x in w)
        if key in seen:
            continue
        seen[key] = len(unique_list)
        unique_list.append([float(x) for x in w])
        first_idx.append(idx)
        key_strs.append(" ".join(f"{x:.{decimals}f}" for x in key))
    return unique_list, first_idx, key_strs

def _expand_with_rela(enabled_groups, cluster_relas_out):
    """enabled_groups: list[int] group idx; cluster_relas_out: list[list[int]] (group idx)"""
    s = set(enabled_groups)
    for g in list(enabled_groups):
        if 0 <= g < len(cluster_relas_out):
            s.update(cluster_relas_out[g])
    return sorted(s)


def parse_cluster_groups_txt(path: str):
    """Parse cluster_groups.txt to get group order exactly as saved.
    Returns: (proto_ids_in_order, proto_vecs_in_order, relas_in_order)
    """
    proto_ids, proto_vecs, relas = [], [], []
    if not Path(path).exists():
        return proto_ids, proto_vecs, relas

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ("prototype:" in line) and ("Query" in line):
                m = re.search(r"Query\s+(\d+)", line)
                if m:
                    proto_ids.append(int(m.group(1)))
                mv = re.search(r"\[(.*?)\]", line)
                if mv:
                    inside = mv.group(1).strip()
                    if inside:
                        parts = [p.strip() for p in inside.split(",")]
                        try:
                            vec = [float(x) for x in parts if x != ""]
                        except Exception:
                            vec = None
                    else:
                        vec = None
                else:
                    vec = None
                proto_vecs.append(vec)

            ls = line.strip()
            if ls.startswith("rela:"):
                nums = re.findall(r"\d+", ls)
                relas.append([int(x) for x in nums])

    return proto_ids, proto_vecs, relas

def save_query_weight_cache_txt(
    save_path: str,
    num_fields: int,
    query_weight_source: str,
    query_weight_list: list,
    sim_matrix: np.ndarray,
    group_global_ids: list,
    group_vectors: list,
    cluster_relas_out: list,
    total_sim_thresh: float,
    rela_sim_thresh: float,
    rela_use_intersect: bool,
    forced_order_way: int,
    search_corr_thresh: float,
    decimals: int = 6
):
    q_unique, q_first_idx, q_keys = dedup_query_weights(query_weight_list, decimals=decimals)
    K = len(group_global_ids)
    Q = len(q_unique)

    with open(save_path, "w") as f:
        f.write("# SmartIndex QueryWeight Cache (for GS-corr fast decision)\n")
        f.write(f"num_fields: {num_fields}\n")
        f.write(f"query_weight_source: {query_weight_source}\n")
        f.write(f"num_query_weights_raw: {len(query_weight_list)}\n")
        f.write(f"num_query_weights_unique: {Q}\n")
        f.write(f"total_sim_thresh: {total_sim_thresh}\n")
        f.write(f"rela_use_intersect: {int(rela_use_intersect)}\n")
        f.write(f"rela_sim_thresh: {rela_sim_thresh}\n")
        f.write(f"forced_order_way: {forced_order_way}\n")
        f.write(f"search_corr_thresh_cached: {search_corr_thresh}\n")
        # f.write(f"search_corr_thresh_cached: {rela_use_intersect}\n") # search_corr_thresh = rela_use_intersect
        f.write(f"dedup_decimals: {decimals}\n")
        f.write(f"num_groups: {K}\n")

        f.write("group_global_ids: " + " ".join(str(int(x)) for x in group_global_ids) + "\n")
        f.write("group_vectors:\n")
        for g, (gid, gv) in enumerate(zip(group_global_ids, group_vectors)):
            gv_str = " ".join(f"{float(x):.{decimals}f}" for x in gv)
            f.write(f"  g{g} gid={int(gid)} vec={gv_str}\n")

        f.write("cluster_relas_out(group_indices):\n")
        for g, relas in enumerate(cluster_relas_out):
            f.write(f"  g{g}: " + " ".join(str(int(x)) for x in relas) + "\n")

        f.write("queries:\n")
        for uid, (vec, src_idx, key_str) in enumerate(zip(q_unique, q_first_idx, q_keys)):
            corrs = [float(sim_matrix[src_idx, int(gid)]) for gid in group_global_ids]
            enabled = [g for g, c in enumerate(corrs) if c >= search_corr_thresh]
            enabled_rela = _expand_with_rela(enabled, cluster_relas_out)

            vec_str = " ".join(f"{float(x):.{decimals}f}" for x in vec)
            corr_str = " ".join(f"{c:.{decimals}f}" for c in corrs)
            en_str = " ".join(str(int(g)) for g in enabled)
            enr_str = " ".join(str(int(g)) for g in enabled_rela)

            f.write(f"  Q{uid} key={key_str}\n")
            f.write(f"    vec: {vec_str}\n")
            f.write(f"    corr_to_groups: {corr_str}\n")
            f.write(f"    enabled_groups@tau={search_corr_thresh}: {en_str}\n")
            f.write(f"    enabled_groups_with_rela@tau={search_corr_thresh}: {enr_str}\n")

    print(f"✅ Saved query-weight cache to {save_path}")


def build_cache_group_view(result: dict,
                           cluster_relas_out: list[list[int]],
                           needInverse: bool) -> tuple[list[int], list[list[float]], list[list[int]]]:
    """Build (group_global_ids, group_vectors, cluster_relas_view) in the EXACT same group order
    as save_cluster_represents(..., needInverse=needInverse) would write.

    Assumptions:
      - result['prototypes'] is a list of (proto_id, weight_vec)
      - cluster_relas_out is list[list[int]] where items are *group indices* (0..K-1) in the ORIGINAL order
        (i.e., matching result['prototypes'] before any inverse).
    """
    prototypes = result.get("prototypes", [])
    K = len(prototypes)
    if K == 0:
        return [], [], []

    if not needInverse:
        order = list(range(K))
        group_ids = [prototypes[i][0] if prototypes[i][0] is not None else -1 for i in order]
        group_vecs = [prototypes[i][1] for i in order]
        relas_view = cluster_relas_out
        return group_ids, group_vecs, relas_view

    # needInverse=True: reverse order and remap group-index relas exactly like save_cluster_represents
    order = list(reversed(range(K)))           # new index j corresponds to old index order[j]
    rev_map = {old: K - 1 - old for old in range(K)}  # old -> new

    group_ids = [prototypes[i][0] if prototypes[i][0] is not None else -1 for i in order]
    group_vecs = [prototypes[i][1] for i in order]

    relas_view = [[] for _ in range(K)]
    for old in range(K):
        new = rev_map[old]
        old_list = cluster_relas_out[old] if old < len(cluster_relas_out) else []
        relas_view[new] = sorted({rev_map[j] for j in old_list if 0 <= j < K})

    return group_ids, group_vecs, relas_view

# --- 1. 读取 base_data_list，准备query_weight_list ---
base_data_list_data = [fun_0817.fvecs_read(path) for path in base_data_list]
num_fields = len(base_data_list_data)
print(f"Loaded {num_fields} fields.")

query_weight_list, query_weight_source = load_query_weights_or_fallback(
    num_fields=num_fields,
    path=use_weight_each_query_txt,
    verbose=True
)
print(f"[useWeight] source={query_weight_source}, num_weights={len(query_weight_list)}")

# - 2026.04.08: 打出所有的query_weight:
print("----- Showing All Workload Weight -------------------------")
for i in range(len(query_weight_list)):
    print(i, ": ", query_weight_list[i])





#  ------------------------
# | 展示sim_matrix
#  ------------------------
def print_sim_matrix(sim_matrix, col_width=10, precision=6):
    """
    sim_matrix : 2D numpy array
    col_width  : 每列宽度（含小数与前导空格）
    precision  : 小数位
    rela_sim_thresh: 阈值，小于该值打印为 0
    """
    indices = [int(x) for x in list(range(sim_matrix.shape[0]))]

    # 头部
    header_pad = " " * (col_width + 2)  # 给行号和 ": " 让位
    col_head   = "".join(f"{j:>{col_width}d}" for j in indices)

    print(header_pad + col_head)

    # 行格式
    idx_fmt = f"{{:>{col_width}d}}: "
    val_fmt = f"{{:>{col_width}.{precision}f}}"

    for i in indices:
        row_vals = "".join(
            val_fmt.format(float(v) if np.isfinite(v) else 0.0)
            for v in (sim_matrix[i, j] for j in indices)
        )
        print(idx_fmt.format(i) + row_vals)

    print("-" * (len(header_pad) + len(col_head)))

#  ------------------------
# | 展示 sim_matrix（低于阈值的元素打印为 0）
#  ------------------------
def print_sim_submatrix(sim_matrix, indices, col_width=10, precision=6, rela_sim_thresh=-1.0):
    """
    sim_matrix : 2D numpy array
    indices    : 需要展示的行/列索引列表（如 S_rep）
    col_width  : 每列宽度（含小数与前导空格）
    precision  : 小数位
    rela_sim_thresh: 阈值，小于该值打印为 0
    """
    indices = [int(x) for x in indices]

    # 头部
    header_pad = " " * (col_width + 2)  # 给行号和 ": " 让位
    col_head   = "".join(f"{j:>{col_width}d}" for j in indices)

    print("---- 展示小部分 sim_matrix -----------------")
    print(header_pad + col_head)

    # 行格式
    idx_fmt = f"{{:>{col_width}d}}: "
    val_fmt = f"{{:>{col_width}.{precision}f}}"

    for i in indices:
        row_vals = "".join(
            val_fmt.format(float(v) if np.isfinite(v) and v >= rela_sim_thresh else 0.0)
            for v in (sim_matrix[i, j] for j in indices)
        )
        print(idx_fmt.format(i) + row_vals)

    print("-" * (len(header_pad) + len(col_head)))







#  ------------------------
# | 计算 distance Pearson Similarity
#  ------------------------
rng = np.random.default_rng(random_state)

num_fields = len(base_data_list_data)
num_weights = len(query_weight_list)
print(f"[MAG] Loaded {num_fields} fields. Num weights: {num_weights}")

# 1) 抽样 base_data
N = len(base_data_list_data[0])
sample_size = min(sample_size, N)
sample_idx = rng.choice(N, size=sample_size, replace=False)
sampled_base_data_list = [field[sample_idx] for field in base_data_list_data]
print(f"[MAG] Sampled base_data shape per field: {[d.shape for d in sampled_base_data_list]}")

# 2) 抽样 query
num_query = min(num_query, sample_size)
query_idx = rng.choice(sample_size, size=num_query, replace=False)

# 3) 预计算每字段欧氏距离 D_j
D_list = fun_0817._precompute_dists_per_field_linear_sum(
    sampled_base_data_list=sampled_base_data_list,
    query_idx=query_idx,
    n_jobs=n_jobs
)
print(f"[MAG] Prepared {len(D_list)} per-field distance mats of shape {D_list[0].shape}")

# 4) 对所有权重生成 d_w = Σ_j w_j * D_j
distance_mats = fun_0817._compute_all_distance_mats_linear_sum(
    query_weight_list=query_weight_list,
    D_list=D_list,
    n_jobs=n_jobs
)
print(f"[MAG] Built {len(distance_mats)} distance mats; example shape: {distance_mats[0].shape}")

# 5) 基于“距离幅度”的权重相似度
# 下面两个参数可调“相似度”的口味：
sim_metric="pearson"  # 'pearson' / 'cosine' / 'nrmse'
sim_center=False
sim_normalize=None
sim_matrix = fun_0817.compute_full_distance_similarity_magnitude(
    distance_mats=distance_mats,
    metric=sim_metric,
    center=sim_center,
    normalize=sim_normalize,
    n_jobs=n_jobs
)
print(f"[MAG] Magnitude-based similarity matrix shape: {sim_matrix.shape}")
print("----- Init sim_matrix: -------------------------")
print_sim_matrix(sim_matrix, col_width=11, precision=6)
# print_sim_submatrix(sim_matrix, list(range(sim_matrix.shape[0])), col_width=11, precision=6)
print("------------------------------------------")






#  ------------------------
# | 根据sim_matrix来做argmax
#  ------------------------
def _validate_sim(sim):
    sim = np.asarray(sim, dtype=float)
    if sim.ndim != 2:
        raise ValueError("sim must be a 2D array")
    n, m = sim.shape
    if n != m:
        # 允许非方阵，但通常代表索引来自列维度
        # 若你只在同一集合内选代表，则请提供方阵或对齐行列的索引空间
        pass
    return sim

def select_k(sim, k, forbid_self=False):
    """
    贪心求解：给定预算 k，最大化 f(S)=sum_i max_{j∈S} sim[i,j]
    返回: S(列表), best_per_row(n维), objective(float)
    """
    sim = _validate_sim(sim)
    n, m = sim.shape
    if k <= 0:
        return [], np.zeros(n), 0.0
    k = min(k, m)

    # 每一行当前被代表覆盖到的最好相似度（初始为0；若你的相似度可为负可改为 -inf）
    best = np.zeros(n, dtype=float)
    chosen = []
    # 为了禁止“自己代表自己”的选法（如 i 不能选为表示 i），可在每轮动态屏蔽
    mask = np.zeros(m, dtype=bool)  # 已选列不再可选

    for _ in range(k):
        # 计算每个候选列的边际增益：将 best 提升为 max(best, sim[:, j]) 的总增量
        gains = np.where(mask, -np.inf, (sim - best[:, None]).clip(min=0).sum(axis=0))

        j_star = int(np.argmax(gains))
        if not np.isfinite(gains[j_star]) or gains[j_star] <= 0:
            # 再选也没增益，提前停止
            break

        chosen.append(j_star)
        mask[j_star] = True
        # 更新每行最佳覆盖
        best = np.maximum(best, sim[:, j_star])

        if forbid_self:
            # 如果不允许“行 i 由列 i 覆盖”，可在这里在下一轮屏蔽对角项影响
            # 实现方式取决于你的业务，这里仅给提示，不默认处理
            pass

    obj = best.sum()/len(query_weight_list)
    return chosen, best, float(obj)


print("### 1.  根据 total_sim_thresh 选择 repre_list(S_k) ################################")
value_list = []
if (max_group == 0):
    max_group = len(query_weight_list)
else:
    max_group = min(len(query_weight_list), max_group)
for k in range(max_group + 1):
    # 假设你已有 sim_matrix (n x n)
    S_k, best_k, obj_k = select_k(sim_matrix, k=k)
    # S_k = sorted(S_k)
    value_list.append(obj_k)
    print("chosen (", k, "):", S_k, "objective:", obj_k)
    if (obj_k > total_sim_thresh):
        final_k = k
        break
print("---- Initial: chosen (", k, "):", S_k, "objective:", obj_k, "----")
print("####################################################################\n\n\n")








print("### 2. 检查是否有field没有被选用，如果有，则补充一组包括所有 ################################")

# 如果 Version 1：----- allCombinWeights ------
result = {"prototypes": [], 'members_per_cluster':[]}  # ✅ 关键修复：先初始化为列表
for w in range(len(S_k)):
    result["prototypes"].append((S_k[w], query_weight_list[S_k[w]]))

fun_0817.supplement_missing_vector(result, base_data_list_data)


# 对sim_matrix增加一行一列，补充其他weight对这一个add_weight下的correlation，注意保持和之前计算sim_matrix时候用一样的sampling和计算方法
# ===== 增量扩充 sim_matrix：只算新增权重与旧权重的相似度 =====
if len(result["prototypes"]) != len(S_k):
    # 1) 取新增的权重（supplement_missing_vector 约定把新增放最后）
    add_weight = result['prototypes'][-1][1]  # list[float], 长度 = num_fields
    query_weight_list.append(add_weight)
    S_k.append(len(sim_matrix))

    # 2) 线性叠加得到新增权重的距离矩阵 add_mat：d_w = Σ_j w_j * D_j
    add_mat = fun_0817._distances_for_weight_linear_sum(D_list, add_weight)
    distance_mats.append(add_mat)  # 维护和距离矩阵列表的一致性

    # 3) 准备向量化工具（与全量计算完全同口径）
    def _make_vecs(mats, upto=None):
        upto = len(mats) if upto is None else upto
        return [
            fun_0817._vectorize_for_similarity(
                M,
                center=(sim_center and sim_metric == "cosine"),
                normalize=sim_normalize
            )
            for M in mats[:upto]
        ]

    # 之前没有缓存过旧向量（vecs_old），此处构建一次（只做一次，不重算旧-旧相似度）
    # 仅对旧的 distance_mats[0 : n_old] 做向量化
    vecs_old = _make_vecs(distance_mats, upto=len(distance_mats) - 1)

    # 新权重的向量化
    v_new = fun_0817._vectorize_for_similarity(
        add_mat,
        center=(sim_center and sim_metric == "cosine"),
        normalize=sim_normalize
    )

    # 4) 计算旧对新（i,new）的相似度（不计算旧-旧）
    # from joblib import Parallel, delayed
    def _sim_to(i):
        val = fun_0817._pair_similarity_magnitude(vecs_old[i], v_new, metric=sim_metric)
        if not np.isfinite(val):
            val = 0.0
        return i, float(val)

    n_old = sim_matrix.shape[0]
    assert n_old == len(vecs_old) == len(distance_mats) - 1, "尺寸不一致，请检查状态"

    pairs = Parallel(n_jobs=n_jobs)(delayed(_sim_to)(i) for i in range(n_old))

    # 5) 扩展相似度矩阵：把旧块复制过来，只补新行/新列
    sim_ext = np.eye(n_old + 1, dtype=np.float32)
    sim_ext[:n_old, :n_old] = sim_matrix
    for i, val in pairs:
        sim_ext[i, n_old] = sim_ext[n_old, i] = val
    sim_matrix = sim_ext

    # 6) 维护缓存，便于后续继续增量
    vecs_old.append(v_new)

    print(f"[MAG] Incrementally extended sim_matrix -> {sim_matrix.shape}")
    print("----- Init sim_matrix: -------------------------")
    print_sim_submatrix(sim_matrix, list(range(sim_matrix.shape[0])), col_width=11, precision=6)
    print("------------------------------------------")


print("---- Initial: chosen (", k, "):", S_k, "----")
print("####################################################################\n\n\n")








print("### 3. 对选出来的 repre 根据 'selectRelaIntersect_weightId(result, S_rep)' 或者 'selectRelaThresh()' 选择每个group的相关部分 ################################")
### 对选出来的repre之间进行排序
def selectRelaThresh(sim_matrix, S_rep, rela_sim_thresh=0.5,
                     print_preview=True):
    """
    在代表集合 S_rep 内，根据阈值构建“相关列表”。

    参数
    ----
    sim_matrix : 2D ndarray (N x N)
    S_rep      : 代表的全局索引列表（如 [14,10,4,...]）
    rela_sim_thresh : 相似度阈值（包含阈值：>=）
    print_preview  : 是否打印子矩阵预览（heatmap 风格）
    full_length    : 返回结构是否扩展到长度 N（其余位置为空列表）

    返回
    ----
    cluster_rela, cluster_rela_reverse
        若 full_length=False：
            - dict[int, list[int]]，仅包含 S_rep 中的键，值为满足阈值的邻居（同在 S_rep）
        若 full_length=True：
            - list[list[int]]，长度 N，其中非 S_rep 位置是 []，S_rep 位置为其邻居列表
    """
    S_rep = list(map(int, S_rep))
    N = sim_matrix.shape[0]
    final_K = len(S_rep)

    if print_preview:
        print("---- 展示小部分 sim_matrix -----------------")
        print_sim_submatrix(sim_matrix, S_rep, col_width=11, precision=6, rela_sim_thresh = rela_sim_thresh)
        print("------------------------------------------")

    # 准备返回容器
    cluster_rela = {i: [] for i in range(N)}
    cluster_rela_reverse = {i: [] for i in range(N)}

    # 只在 S_rep × S_rep 内做判断
    # for a in S_rep:
    #     for b in S_rep:
    #         if a == b:
    #             continue
    #         if sim_matrix[a, b] >= rela_sim_thresh:
    #             cluster_rela[a].append(b)
    #             cluster_rela_reverse[b].append(a)

    for i in range(final_K):
        for j in range(final_K):
            if (i != j and sim_matrix[S_rep[i], S_rep[j]] >= rela_sim_thresh):
                cluster_rela[S_rep[i]].append(S_rep[j])
                cluster_rela_reverse[S_rep[j]].append(S_rep[i])
            

    # # 去重并排序（更稳定易读）
    # def _dedup_sort(container):
    #     if isinstance(container, dict):
    #         for k, v in container.items():
    #             container[k] = sorted(set(v))
    #     else:  # list of lists
    #         for i in range(len(container)):
    #             container[i] = sorted(set(container[i]))
    #     return container

    # cluster_rela = _dedup_sort(cluster_rela)
    # cluster_rela_reverse = _dedup_sort(cluster_rela_reverse)

    # return cluster_rela, cluster_rela_reverse, S_rep
    return cluster_rela, S_rep


# 示例调用
if rela_use_intersect:
    cluster_rela = fun_0817.selectRelaIntersect_weightId(result, S_k)
else:
    # cluster_rela, cluster_rela_reverse, S_k = selectRelaThresh(sim_matrix, S_k, rela_sim_thresh = rela_sim_thresh, print_preview=True)
    cluster_rela, S_k = selectRelaThresh(sim_matrix, S_k, rela_sim_thresh = rela_sim_thresh, print_preview=True)
k = len(S_k)
print("\ncluster_rela:\n", cluster_rela)
# print("\ncluster_rela_reverse:\n", cluster_rela_reverse)
print("####################################################################\n\n\n")








# print("### 4. 对选出来的repre之间进行排序 ################################")
# def order_by_iterative_forward_influence(sim_matrix,
#                                          S_rep,
#                                          rela_sim_thresh=0.2,
#                                          cluster_rela=None,
#                                          rela_use_intersect = True,
#                                          verbose=True):
#     """
#     迭代选择：每步挑选对“未选集合”的阈值化相似度和最大的点。
#     - sim_matrix: (N,N) 对称相似度矩阵
#     - S_rep: 参与排序的全局索引列表（长度 k）
#     - rela_sim_thresh: 小于该阈值的 sim 视为 0（无影响）
#     - cluster_rela: 可选约束。如果提供，则只有 j ∈ cluster_rela[i] 的边才允许计入
#         * dict[int, list[int]] 或 list[list[int]]（长度 N）
#     返回：
#       - S_order: 排序后的全局索引
#       - step_scores: 每一步被选中节点对“当时未选集合”的影响得分
#       - objective: sum(step_scores)（即“前→后”的总影响）
#     """
#     S_rep = list(map(int, S_rep))
#     k = len(S_rep)

#     # 取出 S_rep 子矩阵并做一次阈值化：小于阈值的设为 0；对角也设为 0
#     sub = np.asarray(sim_matrix, dtype=float)[np.ix_(S_rep, S_rep)].copy()
#     np.fill_diagonal(sub, 0.0)
#     if not rela_use_intersect:
#         sub[sub < rela_sim_thresh] = 0.0
#     else:
#         # rela_use_intersect 会 若给了 cluster_rela，再做一层掩码：只允许 i -> j 属于 cluster_rela[i] 的边
#         if cluster_rela is not None:
#             # 构造一个 (k,k) 的 bool 掩码 allowed[i_local, j_local]
#             allowed = np.zeros_like(sub, dtype=bool)
#             # 支持 dict 或 list[list]
#             get_list = (lambda g: cluster_rela.get(g, [])) if isinstance(cluster_rela, dict) \
#                     else (lambda g: cluster_rela[g])
#             gidx = S_rep
#             pos = {g:i for i, g in enumerate(gidx)}
#             for i_g in gidx:
#                 i = pos[i_g]
#                 allowed_js = [j_g for j_g in get_list(i_g) if j_g in pos and j_g != i_g]
#                 for j_g in allowed_js:
#                     allowed[i, pos[j_g]] = True
#             # 把不允许的边置零
#             sub = np.where(allowed, sub, 0.0)
#         else:
#             print("rela_use_intersect, but don't get cluster_rela!!!")
#             # """清空指定txt文件并退出Python程序"""
#             # 1. 以写入模式打开文件会自动清空内容
#             with open(cluster_group_txt, 'w') as f:
#                 pass  # 不写任何内容即可清空

#             print(f"✅ Cleared file: {cluster_group_txt}")
#             # 2. 退出程序
#             sys.exit(0)

#     # 迭代选择
#     remaining = np.ones(k, dtype=bool)
#     S_order_local = []
#     step_scores = []

#     for step in range(k):
#         # 候选集合的行对 “仍未选的列” 的行和
#         cand_idx = np.where(remaining)[0]
#         if cand_idx.size == 0:
#             break
#         # 只统计到“未选”的影响
#         col_mask = remaining
#         # 行和（对未选列求和）
#         scores = sub[cand_idx][:, col_mask].sum(axis=1)

#         # 选最大；平手时按 S_rep 的先后（稳定 argmax）
#         pick_pos = int(np.argmax(scores))
#         pick_local = int(cand_idx[pick_pos])
#         S_order_local.append(pick_local)
#         step_scores.append(float(scores[pick_pos]))

#         if verbose:
#             g = S_rep[pick_local]
#             print(f"[step {step+1}/{k}] pick {g}  influence_to_remaining={scores[pick_pos]:.6f}")

#         # 移除该点（不再作为“未选”）
#         remaining[pick_local] = False

#     # 映射回全局索引
#     S_order = [S_rep[i] for i in S_order_local]
#     objective = float(np.sum(step_scores))

#     if verbose:
#         print(f"[FINAL] objective={objective:.6f}, order={S_order}")

#     return S_order, step_scores, objective

# # 示例调用
# print("---- 选择顺序前的中间态: chosen (", k, "):", S_k, "----")
# S_rep = S_k
# # S_order, scores = order_by_forward_influence(sim_matrix, S_rep=S_k, cluster_rela=cluster_rela, verbose=True)
# # print("\nFinal S_order:", S_order)
# # print("Step scores   :", [f"{s:.6f}" for s in scores])
# S_order, step_scores, obj = order_by_iterative_forward_influence(
#     sim_matrix, S_rep=S_k, rela_sim_thresh = rela_sim_thresh,
#     cluster_rela = cluster_rela, rela_use_intersect = rela_use_intersect,
#     verbose=True
# )
# print("Final order:", S_order)
# print("Step scores:", [f"{x:.6f}" for x in step_scores])
# print("Objective  :", f"{obj:.6f}")
# print("####################################################################\n\n\n")
print("### 4. 根据选出repre的顺序作为构建顺序 ################################")
#  -----------
# ｜ 如果强制要求顺序，（m = 2) 的时候
#  -----------

# 对
def _find_rep_id_by_vec(S_rep, query_weight_list, target_vec, atol=1e-6):
    tv = np.asarray(target_vec, dtype=float)
    for gid in S_rep:
        v = np.asarray(query_weight_list[gid], dtype=float)
        if v.shape == tv.shape and np.allclose(v, tv, atol=atol):
            return int(gid)
    return None

def apply_forced_order_for_2field(S_rep, query_weight_list, way):
    """
    仅在 2-field + reps 包含 [1,0],[0,1],[1,1] 时生效。
    返回新的 S_order；若无法应用则返回 None。
    """
    if way is None or way < 1 or way > 6:
        return None

    a = _find_rep_id_by_vec(S_rep, query_weight_list, [1.0, 0.0])
    b = _find_rep_id_by_vec(S_rep, query_weight_list, [0.0, 1.0])
    c = _find_rep_id_by_vec(S_rep, query_weight_list, [1.0, 1.0])

    if any(x is None for x in (a, b, c)):
        return None

    perms = {
        1: [a, b, c],
        2: [a, c, b],
        3: [b, a, c],
        4: [b, c, a],
        5: [c, a, b],
        6: [c, b, a],
    }
    main = perms[way]

    # 如果未来你有 >3 个 rep（比如补充了 all-ones），就把剩下的按原顺序 append，保证稳定
    rest = [int(x) for x in S_rep if int(x) not in set(main)]
    return main + rest


S_order = list(S_k)

if forced_order_way != -1:
    forced = apply_forced_order_for_2field(S_order, query_weight_list, forced_order_way)
    if forced is None:
        print(f"⚠️ forced_order_way={forced_order_way} requested, but cannot match reps "
              f"[1,0],[0,1],[1,1] in current S_k={S_order}. Keep original order.")
    else:
        print(f"[FORCE] forced_order_way={forced_order_way} applied. "
              f"Old order={S_order} -> New order={forced}")
        S_order = forced

print("Final order:", S_order)
print("####################################################################\n\n\n")






print("### 5. 转换result模式 ################################")
# 根据S_order， cluster_rela等信息整合信息，为下一步调用代码存储信息作准备；
#  ------------------------
# | 根据 S_order / cluster_rela 整合信息，准备保存
#  ------------------------
def build_result_and_relas(S_order, query_weight_list, cluster_rela, use_group_indices=False):
    """
    参数
    ----
    S_order : List[int]
        已排序的代表（全局权重索引）。
    query_weight_list : List[List[float]]
        每个权重配置的向量（与上面的全局索引对齐）。
    cluster_rela : Dict[int, List[int]] 或 List[List[int]]
        “可关联”的邻接信息（使用全局索引表示），通常来自阈值过滤后的关系。
    use_group_indices : bool
        True  -> 将 RELA 转为“组内索引”(0..K-1)
        False -> 将 RELA 保持为“全局权重索引”（推荐，和现有 sim 矩阵的索引一致）

    返回
    ----
    result : dict
        {'prototypes': [(global_id, weight_vec), ...], 'members_per_cluster': [..., ...]}
    cluster_relas_out : List[List[int]]
        每组的 RELA 列表（元素为全局权重索引或组内索引，取决于 use_group_indices）
    """
    # 1) prototypes：按 S_order 的顺序组织 (global_id, weight_vec)
    prototypes = [(gid, query_weight_list[gid]) for gid in S_order]
    final_K = len(prototypes)

    # 2) members_per_cluster：如果此时还没做成员划分，就先留空或各自只包含自己
    #    根据你的管线，如果暂时没有成员分配就用空列表更安全。
    members_per_cluster = [[] for _ in range(final_K)]
    # 也可以选择让每个组先包含自己的代表：
    # members_per_cluster = [[S_order[k]] for k in range(final_K)]

    # 3) cluster_relas_out：长度为 final_K 的列表；每个元素是一组的 RELA 列表
    #    先把 cluster_rela 统一成 "dict<int, List<int]>" 的访问方式
    if isinstance(cluster_rela, dict):
        def neigh(g): return cluster_rela.get(g, [])
    else:
        # list[list] 形式：使用全局索引直接索引
        def neigh(g): return cluster_rela[g]

    # 映射：全局索引 <-> 组内索引
    gidx = list(S_order)
    pos_in_group = {g: k for k, g in enumerate(gidx)}

    cluster_relas_out = []
    for k, g in enumerate(gidx):
        nbrs_global = [j for j in neigh(g) if j in pos_in_group]  # 只保留在本组里的
        nbrs_global = sorted(set(nbrs_global))                     # 去重排序

        if use_group_indices:
            # 转成“组内索引”表达
            nbrs_group = [pos_in_group[j] for j in nbrs_global]
            cluster_relas_out.append(nbrs_group)
        else:
            # 保持“全局权重索引”表达（推荐：与 sim_matrix 的索引一致）
            cluster_relas_out.append(nbrs_global)

    result = {
        "prototypes": prototypes,
        "members_per_cluster": members_per_cluster,
    }
    return result, cluster_relas_out

def relas_to_group_indices(S_order, cluster_rela, include_self=True):
    """
    将 cluster_rela（基于全局ID的邻接）转换为基于组内顺序ID的邻接列表。
    - S_order: List[int]，代表的全局ID，顺序即组内ID（0..K-1）
    - cluster_rela: Dict[int, List[int]] 或 List[List[int]]（全局ID表示）
    - include_self: 是否在每组的关系中包含自己（组内ID）

    返回:
      cluster_relas_out: List[List[int]]，长度 = K
         每个元素是该组的“组内ID”列表（已去重、升序）
    """
    # 兼容 dict 或 list[list] 的访问
    if isinstance(cluster_rela, dict):
        def neigh(g): return cluster_rela.get(g, [])
    else:
        def neigh(g): return cluster_rela[g]

    K = len(S_order)
    pos_in_group = {g: k for k, g in enumerate(S_order)}  # 全局ID -> 组内ID

    cluster_relas_out = []
    for k, g in enumerate(S_order):
        # 取全局邻居，并限制到 S_order 内
        nbrs_global = neigh(g)
        nbrs_group = [pos_in_group[j] for j in nbrs_global if j in pos_in_group]
        if include_self:
            nbrs_group.append(k)  # 包含自己（组内ID）
        # 去重 + 排序（按组内ID升序），得到干净稳定的结果
        nbrs_group = sorted(set(nbrs_group))
        cluster_relas_out.append(nbrs_group)

    return cluster_relas_out


# ===== 组装并保存 =====
# 假设已有：S_order（上一部迭代贪心得到的顺序）、cluster_rela（阈值过滤后的关系，dict 或 list[list]）
result, cluster_relas_out = build_result_and_relas(
    S_order=S_order,
    query_weight_list=query_weight_list,
    cluster_rela=cluster_rela,       # 若不需要方向性/已对称，这里就是对称邻接
    use_group_indices=False          # True: 保存为组内索引；False: 保存为全局权重索引（推荐）
)
print("result:\n", result)
print("cluster_relas_out:\n", cluster_relas_out)
cluster_relas_out = relas_to_group_indices(S_order, cluster_rela, include_self=True)
print("cluster_relas_out:\n", cluster_relas_out)
print("############################################################")



# #  ========== 6. 将结果存入txt =========================================================
print("### 6. 将结果存入txt ################################")
if rela_use_intersect:
    save_path1 = str(Path(out_dir) / f"cluster_group_{当前时间}_Now_withIntersect-total{total_sim_thresh}.txt")
    fun_0817.save_cluster_represents(
        result,
        save_path=save_path1,
        cluster_relas=cluster_relas_out,
        rela_use_intersect=False,
        needInverse=True
    )
else:
    save_path1 = str(Path(out_dir) / f"cluster_group_{当前时间}_Now_withRelaThresh-total{total_sim_thresh}-rela{rela_sim_thresh}.txt")
    fun_0817.save_cluster_represents(
        result,
        save_path=save_path1,
        cluster_relas=cluster_relas_out,
        rela_use_intersect=False,
        needInverse=True
    )

# also save to the fixed path used by C++ (cluster_groups.txt)
cluster_groups_txt_path = cluster_group_txt
fun_0817.save_cluster_represents(
    result,
    save_path=cluster_groups_txt_path,
    cluster_relas=cluster_relas_out,
    rela_use_intersect=False,
    needInverse=True
)

# ---- 2026.01.15: save query-weight cache for fast GS-corr decision ----
# cache_path = str(Path(out_dir) / f"query_weight_cache_{当前时间}_total{total_sim_thresh}_rela{rela_sim_thresh}_tau{search_corr_thresh}.txt")
# cache_path = "/home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/cluster_query_weight_cache_2.txt"
cache_path = "../include/python_file/cluster_query_weight_cache_2.txt"

# ---- 2026.01.15: save query-weight cache for GS-corr ----
# IMPORTANT: group order MUST match cluster_groups.txt exactly.
# Here cluster_groups.txt is written by save_cluster_represents(..., needInverse=True),
# so we must build the cache in the same (reversed) order and remap RELA accordingly.
needInverse_for_groups = True  # keep consistent with the save_cluster_represents call above

group_global_ids_for_cache, group_vectors_for_cache, cluster_relas_for_cache = build_cache_group_view(
    result=result,
    cluster_relas_out=cluster_relas_out,
    needInverse=needInverse_for_groups
)

print("[CHECK] needInverse_for_groups =", needInverse_for_groups)
print("[CHECK] cache group_global_ids:", group_global_ids_for_cache)
print("[CHECK] cache first groups:")
for i in range(min(5, len(group_vectors_for_cache))):
    gid = group_global_ids_for_cache[i]
    vec = group_vectors_for_cache[i]
    print(f"    g{i} gid={gid} vec={vec}")

save_query_weight_cache_txt(
    save_path=cache_path,
    num_fields=num_fields,
    query_weight_source=query_weight_source,
    query_weight_list=query_weight_list,
    sim_matrix=sim_matrix,
    group_global_ids=group_global_ids_for_cache,
    group_vectors=group_vectors_for_cache,
    cluster_relas_out=cluster_relas_for_cache,
    total_sim_thresh=total_sim_thresh,
    rela_sim_thresh=rela_sim_thresh,
    rela_use_intersect=rela_use_intersect,
    forced_order_way=forced_order_way,
    search_corr_thresh=search_corr_thresh,
    decimals=dedup_decimals
)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("############################################################")









# #  ========== 2. 将结果存入txt =========================================================
# import numpy as np
# from pathlib import Path 
# import importlib.util
# from datetime import datetime  # ✅ 用于生成“当前时间”字符串

# # ===== 1.2) 动态加载 fun 模块（文件名含 '-' 无法常规 import，用 importlib 路径加载） =====
# FUN_PATH = Path("/home/mengtong/Link09/LargeIndex/simple_exp/50817_cluster_GMM/ClusterGMM-0817-fun.py")
# assert FUN_PATH.exists(), f"Module file not found: {FUN_PATH}"
# spec_fun = importlib.util.spec_from_file_location("fun_0817", str(FUN_PATH))
# fun_0817 = importlib.util.module_from_spec(spec_fun)
# assert spec_fun.loader is not None
# spec_fun.loader.exec_module(fun_0817)

# # ===== 1.2) 动态加载 “from AnalysePathInfo import load_base_paths as loadBathPath” =====
# AP_PATH = Path("/home/mengtong/Link09/LargeIndex/simple_exp/50817_cluster_GMM/AnalysePathInfo.py")
# assert AP_PATH.exists(), f"Module file not found: {AP_PATH}"
# spec_ap = importlib.util.spec_from_file_location("AnalysePathInfo", str(AP_PATH))
# AnalysePathInfo = importlib.util.module_from_spec(spec_ap)
# assert spec_ap.loader is not None
# spec_ap.loader.exec_module(AnalysePathInfo)
# loadBathPath = AnalysePathInfo.load_base_paths

# # ===== 1.2) 动态加载 “import GenerateClusterWeight_fun as GCW” =====
# GCW_PATH = Path("/home/mengtong/Link09/LargeIndex/simple_exp/50817_cluster_GMM/GenerateClusterWeight_fun.py")
# assert GCW_PATH.exists(), f"Module file not found: {GCW_PATH}"
# spec_gcw = importlib.util.spec_from_file_location("GenerateClusterWeight_fun", str(GCW_PATH))
# GCW = importlib.util.module_from_spec(spec_gcw)
# assert spec_gcw.loader is not None
# spec_gcw.loader.exec_module(GCW)



# # # ####################################################################
# # 逆序（为了贴合写的是逆序的SmartIndex）
# selected_repre = S_order[::-1] 
# # # ####################################################################



# # ===== 3) 主体运行部分 =====
# if __name__ == "__main__":
#     # --- 0. 生成“当前时间”字符串用于输出文件命名（如 20250819-223045） ---
#     当前时间 = datetime.now().strftime("%Y%m%d-%H%M%S")

#     # --- 1. 读取 base_data_list ---
#     base_data_list_data = [fun_0817.fvecs_read(path) for path in base_data_list]
#     num_fields = len(base_data_list_data)
#     print(f"Loaded {num_fields} fields.")

#     # 如果 Version 1：----- allCombinWeights ------
#     for w in range(1, (1 << num_fields)):
#         weights = [(1.0 if (w & (1 << j)) else 0.0) for j in range(num_fields)]
#         query_weight_list.append(weights)

#     result = {"prototypes": [], 'members_per_cluster':[]}  # ✅ 关键修复：先初始化为列表
#     for w in range(len(selected_repre)):
#         result["prototypes"].append((selected_repre[w], query_weight_list[selected_repre[w]]))

#     fun_0817.supplement_missing_vector(result, base_data_list_data)
#     fun_0817.save_cluster_represents(result, save_path=f"/home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/backup_clusterGroups/cluster_group_{当前时间}_withThresh_{total_sim_thresh}.txt")
#     fun_0817.save_cluster_represents(result, save_path=f"/home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/cluster_groups.txt")
#     print("\n-- 【 Hyperparameters 】:")
#     print(f"    total_sim_thresh: {total_sim_thresh}\n")





















# nohup /usr/bin/time -v python3 /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/cluster_Now_distSimThresh_NoOrderRedecide_withCacheAligned_v2.py --total_sim_thresh 0.95 --no-rela_use_intersect --rela_sim_thresh 0.5 > /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/nohup_logs/bash.log60408-clusterSyntheticStand-relaThresh0.5_distSimThresh0.95 2>&1 &

 # -- 默认值 ---------------------------------------
    # sample_size=10000            # 抽样样本数
    # num_query=300                # 每个权重取的查询个数
    # n_jobs=-1                    # 并行核数
    # random_state=42              # 随机种子
    # rela_use_intersect = true (true = --rela_use_intersect; false = --no-rela_use_intersect )
    # total_sim_thresh = 0.95
    # rela_sim_thresh = 0.5
    # ------------------------------------------------



# 2025.12.25: 尝试对 CC1M（m=2）选择所有不同顺序。
# --forced_order_way：1-6
# nohup /usr/bin/time -v python3 /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/cluster_Now_distSimThresh_NoOrderRedecide.py --total_sim_thresh 0.95 --no-rela_use_intersect --rela_sim_thresh -1.0 --forced_order_way 1 > /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/nohup_logs/bash.log51225-clusterCC1M-relaThresh0.3_distSimThresh0.95_ForceOrder 2>&1 &


# 2026.01.15: 尝试记录中间信息。
# nohup /usr/bin/time -v python3 /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/cluster_Now_distSimThresh_NoOrderRedecide_withCacheAligned_v2.py --total_sim_thresh 0.95 --no-rela_use_intersect --rela_sim_thresh 0.1 > /home/mengtong/MyWork/SmartIndex_Final_V1/include/python_file/bash.log_now_2 2>&1 &
