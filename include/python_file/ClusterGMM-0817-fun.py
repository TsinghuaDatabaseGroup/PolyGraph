import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- 1️⃣ 定义读取 fvecs ---
# # 定义ivecs和fvecs的读写函数
# def ivecs_read(fname):
#     a = np.fromfile(fname, dtype='int32')
#     d = a[0]
#     return a.reshape(-1, d + 1)[:, 1:].copy()

# def fvecs_read(fname):
#     print(f"Read from: {fname}")
#     return ivecs_read(fname).view('float32')

def fvecs_read(fname):
    print(f"Read from: {fname}")
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


# ==============================================
# ===== Embedding 计算（完全遵循：先构造 sim 矩阵 → MDS 到 2D）====
# ==============================================

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------- 1) 基元计算：给定权重 → rank matrix ----------
def compute_rank_matrix_one_weight(weight, sampled_base_data_list, query_idx):
    """
    对单个 weight 计算 rank matrix。
    返回: (num_query, sample_size) 的 int32 排名矩阵（0=最近）
    """
    assert len(weight) == len(sampled_base_data_list), \
        f"weight dim {len(weight)} != num_fields {len(sampled_base_data_list)}"
    weighted_fields = [f * weight[j] for j, f in enumerate(sampled_base_data_list)]
    projected = np.concatenate(weighted_fields, axis=1)  # (sample_size, sum_dims)

    ranks_per_query = []
    for q in query_idx:
        dists = np.linalg.norm(projected[q] - projected, axis=1)
        ranks = np.argsort(np.argsort(dists))  # 二次 argsort 得名次
        ranks_per_query.append(ranks.astype(np.int32))
    return np.stack(ranks_per_query, axis=0)  # (num_query, sample_size)


# ---------- 2) 并行计算：所有权重的 rank matrices ----------
def compute_all_rank_matrices(query_weight_list, sampled_base_data_list, query_idx, n_jobs=-1):
    """
    返回: list 长度 = num_weights，每个元素是 (num_query, sample_size) 的 rank 矩阵
    """
    print("Generating rank matrix for all weights...")
    rank_mats = Parallel(n_jobs=n_jobs)(
        delayed(compute_rank_matrix_one_weight)(w, sampled_base_data_list, query_idx)
        for w in tqdm(query_weight_list, desc="Rank matrices")
    )
    print(f"Generated {len(rank_mats)} rank matrices.")
    return rank_mats


# ---------- 3) 全量 Spearman 相似度矩阵（两两权重） ----------
def compute_full_spearman_sim_matrix(rank_matrix_per_weight, num_query, n_jobs=-1):
    """
    输入:
      - rank_matrix_per_weight: list[num_weights]，每个元素形状 (num_query, sample_size)
      - num_query: 实际使用的 query 数
    输出:
      - sim_matrix: (num_weights, num_weights) 的 Spearman 平均相关矩阵
    """
    num_weights = len(rank_matrix_per_weight)
    sim = np.eye(num_weights, dtype=np.float32)

    # 仅计算上三角 (i<j)，并行
    pairs = [(i, j) for i in range(num_weights) for j in range(i + 1, num_weights)]

    def _pair_spearman(i, j):
        rho_list = []
        A = rank_matrix_per_weight[i]
        B = rank_matrix_per_weight[j]
        for q in range(num_query):
            rho, _ = spearmanr(A[q], B[q])
            # 常量向量可能返回 nan，置 0
            rho_list.append(np.nan_to_num(rho, nan=0.0))
        return i, j, float(np.mean(rho_list))

    print("Computing Spearman similarity matrix (pairwise across weights)...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_pair_spearman)(i, j) for (i, j) in tqdm(pairs, desc="Spearman pairs")
    )
    for i, j, val in results:
        sim[i, j] = val
        sim[j, i] = val

    return sim


# ---------- 4) MDS 到 2D（预计算距离：1 - sim）若失败可选用 PCA 退化 ----------
def mds_2d_from_similarity(sim_matrix, random_state=42, fallback_to_pca=True):
    """
    按照既定逻辑：对 (1 - sim_matrix) 作为 precomputed dissimilarity 做 MDS 到 2D。
    若 MDS 失败且 fallback_to_pca=True，则对 sim_matrix 行向量做 PCA(2D) 作为退路。
    返回: feature_2d, method_used ∈ {"MDS", "PCA"}
    """
    n = sim_matrix.shape[0]
    diss = 1.0 - sim_matrix  # Spearman ∈ [-1,1] → 距离 ∈ [0,2]
    try:
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=random_state,
            n_init=4,
            max_iter=300,
            normalized_stress='auto'
        )
        feature_2d = mds.fit_transform(diss)
        if not np.all(np.isfinite(feature_2d)):
            raise ValueError("MDS result has NaN/Inf")
        print("[DimReduce] Used MDS(precomputed) → 2D.")
        return feature_2d, "MDS"
    except Exception as e:
        if not fallback_to_pca:
            raise
        print(f"[DimReduce] MDS failed ({type(e).__name__}: {e}). Fallback to PCA.")
        # 退化：以 sim 行向量作为特征做 PCA(2D)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        feature_2d = pca.fit_transform(sim_matrix)
        return feature_2d, "PCA"


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def cluster_on_2d_with_gmm_0817(feature_2d, query_weight_list,
                           K_max=10, random_state=42, threshold=0.3, show_plot=True):
    """
    在 2D 特征上用 GMM（BIC 选 K），并绘制：
      1) 2D 聚类散点（按主簇着色，标中心）
      2) 样本级树状图（递归染色：分支与叶子颜色与散点一致），并画出 "best K" 的水平切分线
      3) BIC vs K 折线（红线标出 best K）
    返回：包含 K、bic_list、proba、dominant_label、combination_to_indexlist、centers_2d、members_per_cluster 等
    """
    n = feature_2d.shape[0]
    K_range = list(range(1, min(K_max, n) + 1))

    # --- GMM + BIC 选 K ---
    lowest_bic = np.inf
    best_gmm = None
    bic_list = []

    print("\nSweep K by BIC (GMM on 2D):")
    for K in K_range:
        gmm = GaussianMixture(n_components=K, random_state=random_state)
        gmm.fit(feature_2d)
        bic = gmm.bic(feature_2d)
        bic_list.append(bic)
        print(f"  K={K:2d}, BIC={bic:.2f}")
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    best_K = best_gmm.n_components
    print(f"\nSelected best_K = {best_K} with BIC = {lowest_bic:.2f}")

    proba = best_gmm.predict_proba(feature_2d)  # (n, best_K)
    dominant_label = np.argmax(proba, axis=1)   # (n,)
    centers_2d = best_gmm.means_               # (best_K, 2)

    # 阈值多归属（可多标签）
    combination_to_indexlist = {
        i: [j for j in range(best_K) if proba[i, j] >= threshold] for i in range(n)
    }

    print("\nQuery → probability per cluster:")
    for i in range(n):
        probs = ", ".join([f"{proba[i, j]:.3f}" for j in range(best_K)])
        print(f"  Query {i}: [{probs}] → belong: {combination_to_indexlist[i]}")

    # 原型（以 2D 中心到成员最近者）
    print("\n=== Cluster Summary ===")
    prototypes = []
    members_per_cluster = []
    for j in range(best_K):
        member_ids = [i for i in range(n) if j in combination_to_indexlist[i]]
        members_per_cluster.append(member_ids)
        if len(member_ids) > 0:
            dists = cdist([centers_2d[j]], feature_2d[member_ids])[0]
            prototype_idx = member_ids[int(np.argmin(dists))]
            prototype_val = query_weight_list[prototype_idx] if query_weight_list is not None else None
        else:
            prototype_idx = None
            prototype_val = None
        prototypes.append((prototype_idx, prototype_val))

        print(f"\nCluster {j}:")
        print(f"  center (feature2D): {centers_2d[j]}")
        if prototype_idx is not None:
            print(f"  prototype: Query {prototype_idx} → {prototype_val}")
        else:
            print("  prototype: None")
        print(f"  members: {member_ids[:20]}{' ...' if len(member_ids) > 20 else ''}")

    # --- 三联图绘制 ---
    if show_plot:
        fig, axs = plt.subplots(1, 3, figsize=(21, 6))
        ax_scatter, ax_dendro, ax_bic = axs
        cmap = plt.cm.get_cmap('tab10', best_K)

        # 1) 2D 聚类散点
        for j in range(best_K):
            idxs = np.where(dominant_label == j)[0]
            ax_scatter.scatter(feature_2d[idxs, 0], feature_2d[idxs, 1],
                               color=cmap(j), label=f"Cluster {j}", s=30)
        # 显示多归属（可选，避免过密也可注释掉）
        for i, (x, y) in enumerate(feature_2d):
            ax_scatter.text(x, y, f"{i}:{combination_to_indexlist[i]}",
                            fontsize=8, ha='center', va='center')
        # 中心
        ax_scatter.scatter(centers_2d[:, 0], centers_2d[:, 1],
                           marker='X', s=140, color='black', label='Center')
        ax_scatter.set_title(f"2D Clusters (GMM, K={best_K})")
        ax_scatter.legend(loc='best')

        # 2) 样本级树状图（递归染色，让分支 & 叶子颜色与散点一致）
        Z = linkage(feature_2d, method='ward')

        # 递归着色：若某子树下所有叶子同簇，则该分支染为该簇色；否则灰色
        leaf_colors = {i: int(dominant_label[i]) for i in range(n)}
        cluster_map = {}

        def _get_cluster(node_id):
            if node_id < n:
                return leaf_colors[node_id]
            l = int(Z[node_id - n, 0])
            r = int(Z[node_id - n, 1])
            lc = _get_cluster(l)
            rc = _get_cluster(r)
            cid = lc if lc == rc else -1
            cluster_map[node_id] = cid
            return cid

        _get_cluster(n + Z.shape[0] - 1)

        def link_color_func(node_id):
            cid = cluster_map.get(node_id, -1)
            return 'grey' if cid == -1 else to_hex(cmap(cid))

        dendro = dendrogram(
            Z,
            ax=ax_dendro,
            labels=[str(i) for i in range(n)],
            link_color_func=link_color_func,
            color_threshold=None,  # 我们自己处理颜色
            no_labels=False
        )
        ax_dendro.set_title("Dendrogram (colors = GMM clusters)")

        # 给叶子文字上色，保持与散点一致
        for lbl in ax_dendro.get_xmajorticklabels():
            idx = int(lbl.get_text())
            lbl.set_color(cmap(dominant_label[idx]))

        # 根据 best K 画“切分位置”水平线
        # 标准做法：介于第 (n-best_K) 与 (n-best_K-1) 次合并距离之间
        if n > best_K:
            heights = Z[:, 2]
            # 当我们想要 K 个簇时，水平线可放在第 (n-K) 次合并的高度和前一次之间
            cut_height = (heights[-best_K] + heights[-best_K - 1]) / 2 if (n - best_K - 1) >= 0 else heights[-best_K] - 1e-6
            ax_dendro.axhline(cut_height, color='red', linestyle='--', label=f'cut @K={best_K}')
            ax_dendro.legend(loc='best')

        # 3) BIC vs K
        ax_bic.plot(K_range, bic_list, marker='o')
        ax_bic.axvline(best_K, color='red', linestyle='--', label=f'Best K={best_K}')
        for K, bic in zip(K_range, bic_list):
            ax_bic.annotate(f"{bic:.1f}", (K, bic), textcoords="offset points", xytext=(0,6),
                            ha='center', fontsize=9)
        ax_bic.set_xlabel('K (number of clusters)')
        ax_bic.set_ylabel('BIC')
        ax_bic.set_title('BIC vs K')
        ax_bic.legend(loc='best')

        plt.tight_layout()
        plt.show()

    return {
        "K": best_K,
        "bic_list": bic_list,
        "proba": proba,
        "dominant_label": dominant_label,
        "combination_to_indexlist": combination_to_indexlist,
        "centers_2d": centers_2d,
        "feature_2d": feature_2d,
        "prototypes": prototypes,
        "members_per_cluster": members_per_cluster,
    }

# ==============================================
# ===== 🟢 实验入口函数（模块化 + 并行版本） =====
# ==============================================

def run_experiment_MDS(
    base_data_list_data,          # list[num_fields]，每个是 (N, dim_f) 的 np.ndarray
    query_weight_list,            # (num_weights, num_fields)
    sample_size=10000,            # 抽样样本数
    num_query=300,                # 每个权重取的查询个数
    n_jobs=-1,                    # 并行核数
    random_state=42,              # 随机种子
    K_max=10,                     # GMM 选 K 的上限
    threshold=0.3,                # 多归属阈值
    show_plot=True,               # 是否绘图
    mds_fallback_to_pca=True,     # 若 MDS 失败是否退回 PCA
):
    """
    完整流程（完全对齐你给出的单脚本逻辑）：
      1) 抽样 base_data_list_data
      2) 对每个权重计算 rank matrix
      3) 计算所有权重两两之间的 Spearman 相似度矩阵 sim_matrix
      4) 用 MDS(precomputed) 对 (1 - sim_matrix) 降到 2D（失败可选退 PCA）
      5) 在 2D 上用 GMM + BIC 选 K，输出软聚类与可视化
    """
    rng = np.random.default_rng(random_state)

    num_fields = len(base_data_list_data)
    num_weights = len(query_weight_list)
    print(f"Loaded {num_fields} fields. Num weights: {num_weights}")

    # 2️⃣ 抽样 base_data
    N = len(base_data_list_data[0])
    sample_size = min(sample_size, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    sampled_base_data_list = [field[sample_idx] for field in base_data_list_data]
    print(f"Sampled base_data shape per field: {[d.shape for d in sampled_base_data_list]}")

    # 3️⃣ 抽样 query
    num_query = min(num_query, sample_size)
    query_idx = rng.choice(sample_size, size=num_query, replace=False)

    # 4️⃣ 计算所有 rank matrices（并行）
    rank_matrix_per_weight = compute_all_rank_matrices(
        query_weight_list=query_weight_list,
        sampled_base_data_list=sampled_base_data_list,
        query_idx=query_idx,
        n_jobs=n_jobs
    )

    # 5️⃣ Spearman 相似度矩阵（两两权重）
    sim_matrix = compute_full_spearman_sim_matrix(
        rank_matrix_per_weight=rank_matrix_per_weight,
        num_query=num_query,
        n_jobs=n_jobs
    )
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    # 6️⃣ MDS(precomputed) → 2D（失败可退 PCA）
    feature_2d, red_method = mds_2d_from_similarity(
        sim_matrix, random_state=random_state, fallback_to_pca=mds_fallback_to_pca
    )

    # 7️⃣ 在 2D 上做 GMM + BIC 选 K，并可视化
    result = cluster_on_2d_with_gmm_0817(
        feature_2d=feature_2d,
        query_weight_list=np.asarray(query_weight_list),
        K_max=K_max,
        random_state=random_state,
        threshold=threshold,
        show_plot=show_plot
    )

    # 附加有助于复现实验的信息
    result.update({
        "sim_matrix": sim_matrix,
        "reduction_method": red_method,
        "sample_idx": sample_idx.tolist(),
        "query_idx": query_idx.tolist(),
    })
    return result





# ==============================================
# ===== 对于result的补足 =====
# ==============================================

def supplement_missing_vector(result, base_data_list_data):
    """
    将representative weight configurations中始终为0的列补充一个新的综合组。
    """
    prototypes = result['prototypes']
    final_K = len(prototypes)
    num_fields = len(base_data_list_data)
    print(prototypes)

    # === 找missing vectors
    missing = num_fields
    occured = np.zeros(num_fields, dtype=int)
    for j in range(num_fields):
        for i in range(final_K):
            if (prototypes[i][1][j] != 0):
                occured[j] = 1
                missing -= 1
                break

    print(missing)
    # === 要补充的那组
    if (missing):
        print("Exist missing vector fields..")
        missing_vec_weight = np.zeros(num_fields, dtype=float)
        total_norm = 0.0
        for i in range(num_fields):
            if (occured[i] == 0):
                missing_vec_weight[i] = np.max( np.linalg.norm(base_data_list_data[i], axis=1) )
                # print(f"missing_vec_weight[{i}] = {missing_vec_weight[i]}")
                total_norm += missing_vec_weight[i]
        for i in range(num_fields):
            if (occured[i] == 0):
                # print(f"missing_vec_weight_FINALUSED[{i}] = total_norm/missing_vec_weight[i] = {total_norm}/{missing_vec_weight[i]} = {total_norm/missing_vec_weight[i]}")
                missing_vec_weight[i] = total_norm/missing_vec_weight[i]
                
                

        result['prototypes'].append((None, missing_vec_weight))
        result['members_per_cluster'].append([])




# ==============================================
# ===== 存储cluster之后得到的representations 和 related groups =====
# ==============================================

import numpy as np
from scipy.spatial.distance import cdist

def selectRelaIntersect(result):
    prototypes = result['prototypes']
    members_per_cluster = result['members_per_cluster']
    final_K = len(prototypes)
    # === 对每个组找 rela
    cluster_relas = []
    for i in range(final_K):
        rela = []
        for j in range(final_K):
            proto_i = prototypes[i][1]
            # proto_j = [1] * num_fields if j == full_ones_idx else cluster_prototypes[j]
            proto_j = prototypes[j][1]
            overlap = any((a == 1 and b == 1) for a, b in zip(proto_i, proto_j))
            if overlap:
                rela.append(j)
        cluster_relas.append(sorted(set(rela)))
    return cluster_relas

def selectRelaIntersect_weightId(result, S_rep):
    prototypes = result['prototypes']
    members_per_cluster = result['members_per_cluster']
    final_K = len(prototypes)
    # === 对每个组找 rela (# 得到的是组内ID的 list[list])
    cluster_rela_local = []
    for i in range(final_K):
        rela = []
        for j in range(final_K):
            proto_i = prototypes[i][1]
            # proto_j = [1] * num_fields if j == full_ones_idx else cluster_prototypes[j]
            proto_j = prototypes[j][1]
            overlap = any((a == 1 and b == 1) for a, b in zip(proto_i, proto_j))
            if overlap:
                rela.append(j)
        cluster_rela_local.append(sorted(set(rela)))

        # 转全局ID的 dict[int -> List[int]]
        cluster_rela = {g: [] for g in S_rep}
        for i_local, nbrs in enumerate(cluster_rela_local):
            i_g = S_rep[i_local]
            for j_local in nbrs:
                j_g = S_rep[j_local]
                if j_g != i_g:
                    cluster_rela[i_g].append(j_g)
    return cluster_rela

def save_cluster_represents(
    result,
    save_path="cluster_group_protoAndRela.txt",
    cluster_relas=None,                 # ✅ 改为 None
    rela_use_intersect=True,
    rela_sim_thresh=0.2,                # 预留给内部计算（如有）
    needInverse=False,
    rela_is_group_index=True,           # ✅ 明确 RELA 的索引体系：组内ID(默认) / 全局ID
    preview_print=True                  # ✅ 控制是否打印
):
    """
    将聚类代表与关系写出。

    参数：
      - result: 包含
          * 'prototypes': [(global_id or None, weight_vec), ...]
          * 'members_per_cluster': [list[int], ...]
      - cluster_relas: List[List[int]] (K 个列表)。元素为组内ID或全局ID，取决于 rela_is_group_index
      - needInverse: 反序写出（K-1 -> 0）
      - rela_is_group_index: True 表示 RELA 中存的是组内ID (0..K-1)；False 表示全局ID
      - preview_print: 是否打印预览
    """
    prototypes = result.get('prototypes', [])
    members_per_cluster = result.get('members_per_cluster', [])
    final_K = len(prototypes)

    # === 只在没有传入 cluster_relas 时，按配置决定是否内部计算 
    if ( cluster_relas is None and rela_use_intersect ):
        cluster_relas = selectRelaIntersect(result)

    # === 反序视图 & RELA 重映射（仅当 needInverse=True）
    if needInverse:
        rev_order = list(reversed(range(final_K)))         # 新顺序：K-1..0
        rev_map = {old: final_K - 1 - old for old in range(final_K)}

        prototypes_view = [prototypes[i] for i in rev_order]
        members_view = [members_per_cluster[i] if i < len(members_per_cluster) else [] for i in rev_order]

        if rela_is_group_index:
            # 组内ID需要重映射到“反序后的组内ID”
            cluster_relas_view = [[] for _ in range(final_K)]
            for old in range(final_K):
                new = rev_map[old]
                old_list = cluster_relas[old] if old < len(cluster_relas) else []
                # 仅保留合法 [0..K-1]，并映射
                remapped = sorted({rev_map[j] for j in old_list if 0 <= j < final_K})
                cluster_relas_view[new] = remapped
        else:
            # 全局ID无需映射，只随组顺序重排
            cluster_relas_view = [cluster_relas[i] if i < len(cluster_relas) else [] for i in rev_order]

        # 预览
        if preview_print:
            print("\n=== Final Cluster Summary (REVERSED ORDER) ===")
            for j, k_old in enumerate(rev_order):
                print(f"\nCluster {j}  (was old idx: {k_old})")
                gid, wvec = prototypes_view[j]
                if gid is not None:
                    print(f"  prototype: Query {gid} → {wvec}")
                else:
                    print(f"  prototype: None(Append) → {wvec}")
                if j < len(members_view):
                    mems = members_view[j]
                    print(f"  members: {mems[:20]}{' ...' if len(mems) > 20 else ''}")
                print(f"  rela: {cluster_relas_view[j]}")

        # 写文件（反序）
        with open(save_path, 'w') as f:
            f.write(f"{final_K}\n")
            for j in range(final_K):
                proto = " ".join(map(str, prototypes_view[j][1]))
                members = " ".join(map(str, members_view[j])) if j < len(members_view) else ""
                rela = " ".join(map(str, cluster_relas_view[j])) if j < len(cluster_relas_view) else ""
                f.write(f"# Group {j}\n")
                f.write(f"PROTO: {proto}\n")
                f.write(f"MEMBERS: {members}\n")
                f.write(f"RELA: {rela}\n")
        print(f"✅ Saved (reversed) cluster prototype info to {save_path}")
        return

    # === 正常顺序视图
    prototypes_view = prototypes
    members_view = members_per_cluster
    cluster_relas_view = cluster_relas

    # 预览
    if preview_print:
        print("\n=== Final Cluster Summary ===")
        for j in range(final_K):
            print(f"\nCluster {j}:")
            gid, wvec = prototypes_view[j]
            if gid is not None:
                print(f"  prototype: Query {gid} → {wvec}")
            else:
                print(f"  prototype: None(Append) → {wvec}")
            if j < len(members_view):
                mems = members_view[j]
                print(f"  members: {mems[:20]}{' ...' if len(mems) > 20 else ''}")
            print(f"  rela: {cluster_relas_view[j] if j < len(cluster_relas_view) else []}")

    # 写文件（正序）
    with open(save_path, 'w') as f:
        f.write(f"{final_K}\n")
        for k in range(final_K):
            proto = " ".join(map(str, prototypes_view[k][1]))
            members = " ".join(map(str, members_view[k])) if k < len(members_view) else ""
            rela = " ".join(map(str, cluster_relas_view[k])) if k < len(cluster_relas_view) else ""
            f.write(f"# Group {k}\n")
            f.write(f"PROTO: {proto}\n")
            f.write(f"MEMBERS: {members}\n")
            f.write(f"RELA: {rela}\n")
    print(f"✅ Saved cluster prototype info to {save_path}")








# ========== 🔵 2025.09.20 追加功能：距离幅度版（不改动上面任何已有函数） ==========

from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

# 预计算：每个 field 的欧氏距离（查询 vs 全体样本）
def _precompute_dists_per_field_linear_sum(sampled_base_data_list, query_idx, n_jobs=-1):
    """
    返回: list[D_j]，每个 D_j 形状 (num_query, sample_size)，
          D_j[q, o] = || x_q^{(j)} - x_o^{(j)} ||_2
    """
    def _one_field(field_mat):
        Q = field_mat[query_idx]                      # (num_query, d_j)
        Dj = cdist(Q, field_mat, metric='euclidean')  # float64
        return Dj.astype(np.float32, copy=False)

    return Parallel(n_jobs=n_jobs)(
        delayed(_one_field)(fld) for fld in sampled_base_data_list
    )

# 按权重线性求和：d_w = Σ_j w_j * D_j
def _distances_for_weight_linear_sum(D_list, weight):
    assert len(weight) == len(D_list), "weight 维度与字段数不一致"
    Dw = np.zeros_like(D_list[0], dtype=np.float32)
    for wj, Dj in zip(weight, D_list):
        if wj != 0:
            Dw += (wj * Dj)
    return Dw  # (num_query, sample_size)

def _compute_all_distance_mats_linear_sum(query_weight_list, D_list, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(_distances_for_weight_linear_sum)(D_list, w) for w in query_weight_list
    )

# 距离“幅度敏感”的权重相似度矩阵（基于 Pearson / Cosine / NRMSE）
def _vectorize_for_similarity(mat, center=False, normalize=None, eps=1e-12):
    v = mat.reshape(-1).astype(np.float64, copy=False)
    if center:
        v = v - v.mean()
    if normalize == "l2":
        nrm = np.linalg.norm(v) + eps
        v = v / nrm
    return v

def _pair_similarity_magnitude(v1, v2, metric="pearson", eps=1e-12):
    if metric == "pearson":
        v1c, v2c = v1 - v1.mean(), v2 - v2.mean()
        denom = (np.linalg.norm(v1c) * np.linalg.norm(v2c)) + eps
        return float(np.dot(v1c, v2c) / denom)
    elif metric == "cosine":
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + eps
        return float(np.dot(v1, v2) / denom)
    elif metric == "nrmse":
        rmse = np.sqrt(np.mean((v1 - v2) ** 2))
        scale = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0 + eps
        return float(1.0 - rmse / scale)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_full_distance_similarity_magnitude(distance_mats, metric="pearson",
                                              center=False, normalize=None, n_jobs=-1):
    """
    输入:
      - distance_mats: list[num_weights]，每个元素形状 (num_query, sample_size)，为 d_w = Σ_j w_j * D_j
    输出:
      - sim_matrix_mag: (num_weights, num_weights)，“幅度敏感”的相似度
    参数:
      - metric: 'pearson'(默认) / 'cosine' / 'nrmse'
      - center: 对 cosine 是否先中心化（pearson 自带中心化）
      - normalize: 对向量做 'l2' 归一化或 None
    """
    num_weights = len(distance_mats)
    sims = np.eye(num_weights, dtype=np.float32)
    vecs = [
        _vectorize_for_similarity(M, center=(center and metric == "cosine"), normalize=normalize)
        for M in distance_mats
    ]
    pairs = [(i, j) for i in range(num_weights) for j in range(i + 1, num_weights)]

    def _one(i, j):
        val = _pair_similarity_magnitude(vecs[i], vecs[j], metric=metric)
        if not np.isfinite(val):
            val = 0.0
        return i, j, float(val)

    results = Parallel(n_jobs=n_jobs)(delayed(_one)(i, j) for (i, j) in pairs)
    for i, j, val in results:
        sims[i, j] = sims[j, i] = val
    return sims

# 与原管线并存的新入口：基于“距离幅度”的相似度 + MDS + GMM
def run_experiment_MDS_magnitude(
    base_data_list_data,
    query_weight_list,
    sample_size=10000,
    num_query=300,
    n_jobs=-1,
    random_state=42,
    K_max=10,
    threshold=0.3,
    show_plot=True,
    mds_fallback_to_pca=True,
    # 下面两个参数可调“相似度”的口味：
    sim_metric="pearson",   # 'pearson' / 'cosine' / 'nrmse'
    sim_center=False,
    sim_normalize=None,
):
    """
    与 run_experiment_MDS 类似，但相似度来自“加权欧氏距离的幅度相似性”。
    关键定义：d_w(q,o) = Σ_j w_j * ||x_q^(j) - x_o^(j)||_2
    """
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
    D_list = _precompute_dists_per_field_linear_sum(
        sampled_base_data_list=sampled_base_data_list,
        query_idx=query_idx,
        n_jobs=n_jobs
    )
    print(f"[MAG] Prepared {len(D_list)} per-field distance mats of shape {D_list[0].shape}")

    # 4) 对所有权重生成 d_w = Σ_j w_j * D_j
    distance_mats = _compute_all_distance_mats_linear_sum(
        query_weight_list=query_weight_list,
        D_list=D_list,
        n_jobs=n_jobs
    )
    print(f"[MAG] Built {len(distance_mats)} distance mats; example shape: {distance_mats[0].shape}")

    # 5) 基于“距离幅度”的权重相似度
    sim_matrix_mag = compute_full_distance_similarity_magnitude(
        distance_mats=distance_mats,
        metric=sim_metric,
        center=sim_center,
        normalize=sim_normalize,
        n_jobs=n_jobs
    )
    print(f"[MAG] Magnitude-based similarity matrix shape: {sim_matrix_mag.shape}")

    # 6) 用原有的 MDS(precomputed) → 2D（直接复用已有函数）
    feature_2d, red_method = mds_2d_from_similarity(
        sim_matrix_mag, random_state=random_state, fallback_to_pca=mds_fallback_to_pca
    )

    # 7) 在 2D 上 GMM + BIC 选 K，并可视化（直接复用已有函数）
    result = cluster_on_2d_with_gmm_0817(
        feature_2d=feature_2d,
        query_weight_list=np.asarray(query_weight_list),
        K_max=K_max,
        random_state=random_state,
        threshold=threshold,
        show_plot=show_plot
    )

    # 附加：便于复现/下游使用
    result.update({
        "sim_matrix_magnitude": sim_matrix_mag,
        "reduction_method": red_method,
        "sample_idx": sample_idx.tolist(),
        "query_idx": query_idx.tolist(),
        "sim_metric": sim_metric,
        "sim_center": sim_center,
        "sim_normalize": sim_normalize,
        "distance_definition": "d_w(q,o) = sum_j w_j * ||x_q^(j) - x_o^(j)||_2",
    })
    return result
