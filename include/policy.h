#ifndef XMT_POLICY_H
#define XMT_POLICY_H

namespace xmt {
    enum TYPE {
        // --- Distacne Type 🗑️： 我只支持一种，这些最终可以去掉
        DIST_EUCLIDEAN, DIST_COS, DIST_COS_SIMILARITY,

        // --- Index type:
        INDEX_SMART,
        INDEX_HNSW_FUSION,
        INDEX_VAMANA_BASELINE, INDEX_ORACLE_BASELINE,
        //---- 这个按理说应该被删掉：
        INDEX_VAMANA, INDEX_HNSW,

        // --- Edge-group 划分方式：
        CLUSTER_GROUP, GROUP_EQU_NO_TOTAL, GROUP_FUSION, GROUP_ALL_WEIGHT,

        // --- Graph Initialization type:
        INIT_RAND,
        INIT_HNSW_FUSION,

        // --- Connectivity Enforcement
        CONNECT_RELA,


        // --- Hypereparameter 划分方式：
        PARAM_EQUAL, 




        // === Search ===
        // --- Search Entry Point Type:
        SEARCH_ENTRY_NONE_FUSION, SEARCH_ENTRY_CENTROID,

        // --- Search Router Type:
        ROUTER_HNSW_FUSION,
        ROUTER_GREEDY, ROUTER_GREEDY_ALL_INDEX,

        // --- How to set L_Search:
        L_SEARCH_ASSIGN_FALLBACKINTERSECT, L_SEARCH_ASSIGN_ALL_INDEX, L_SEARCH_ASSIGN_INTERSECT,

        // L_RECALL_SEARCH_CONTROL, 
        L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT, 
        L_RECALL_SEARCH_CONTROL_ALL_INDEX, L_RECALL_SEARCH_CONTROL_INTERSECT, L_RECALL_SEARCH_CONTROL_EXACT_REPRE, 
        L_RECALL_SEARCH_CONTROL_QUERY_SELECT_REPRE,


        // --- How to get Query Weights:
        LOAD_WEIGHT, 
        ALL_WEIGHT, LOADED_ALL_WEIGHT, 
    };
}

#endif //XMT_POLICY_H
