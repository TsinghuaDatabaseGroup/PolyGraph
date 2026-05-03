#include "component.h"


// ======================================
// Search Flow
// ======================================

namespace xmt {
    void ComponentSearchFlowLoadWeight_smart::prepareForSearch_smart_exactRepre(unsigned query, std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                                std::vector<unsigned> &checkEps)
    {
        std::vector<float> weight = smart_index->getSearchWeight_q(query);
        std::set<int> needIndeces_set;

        needCalField.clear();
        needIndeces.clear();
        checkEps.clear();
        for (int i = 0; i < weight.size(); i++)
        {
            if (weight[i])
            {
                needCalField.emplace_back(i);
            }
        }

        // ## 2. 判断用哪些index
        // ### --- 必须weight == represent ---
        {
            // ### --- 必须weight == represent ---
            for (int ind = 0; ind < smart_index->getGroupNum(); ind++)
            {
                auto &fElements = smart_index->getGroupRepreList()[ind];

                bool skip = false;
                for (int i = 0; i < fElements.size(); i++)
                {
                    if (weight[i] != fElements[i])
                    {
                        skip = true;
                        break;
                    }
                }
                if (!skip)
                {
                    if (query == 0)
                    {
                        std::cout << "weight: ";
                        for (int i = 0; i < fElements.size(); i++)
                        {
                            std::cout << weight[i] << ", ";
                        }
                        std::cout << "      ; fElements: ";
                        for (int i = 0; i < fElements.size(); i++)
                        {
                            std::cout << fElements[i] << ", ";
                        }
                        std::cout << std::endl;
                    }
                    needIndeces_set.insert(ind);
                    checkEps.emplace_back(smart_index->each_ep_[ind]);
                }
            }
            needIndeces.assign(needIndeces_set.begin(), needIndeces_set.end());

            // ## show needIndeces
            if (query == 0)
            {
                std::cout << "- needIndeces: ";
                for (auto ind : needIndeces)
                {
                    std::cout << ind << ", ";
                }
                std::cout << std::endl;
                std::cout << "- needIndeces end" << std::endl;
            }
        }
    }

    void ComponentSearchFlowLoadWeight_smart::prepareForSearch_smart_SavedAGS_FallbackIntersect(unsigned query, std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                     std::vector<unsigned> &checkEps)
    {
        std::vector<float> weight = smart_index->getSearchWeight_q(query);

        // 在getUniqueWorkloadFlat()中找weight是否出现过
        unsigned weight_id = 0;
        bool has_diff = false;
        unsigned offset = 0;
        std::vector<float> &weights_flat = smart_index->getUniqueWorkloadFlat();
        for (; weight_id < smart_index->getUniqueWorkloadNum(); weight_id++) {
            has_diff = false;
            for (int i = 0; i < smart_index->getFieldNum(); i++) {
                if (weight[i] != weights_flat[offset+i]) {
                    has_diff = true;
                    break;
                }
            }
            if (!has_diff) {break;}
            offset += smart_index->getFieldNum();
        }

        if (!has_diff) 
        {
            std::set<int> needCalField_set;
            needCalField.clear();
            needIndeces.clear();
            for (int i = 0; i < weight.size(); i++)
            {
                if (weight[i])
                {
                    needCalField.emplace_back(i);
                    needCalField_set.insert(i);
                }
            }

            for (unsigned pos = smart_index->getEnableGroupRecordOffset()[weight_id]; pos < smart_index->getEnableGroupRecordOffset()[weight_id+1]; pos++) {
                needIndeces.emplace_back(smart_index->getEnabledGroupFlat()[pos]);
            }

        }
        else {
            // FallBack to Intersect
            prepareForSearch_smart_intersect(query, needCalField, needIndeces, checkEps);
        }

        // FallBack to Intersect
        if (needIndeces.size() == 0) { prepareForSearch_smart_intersect(query, needCalField, needIndeces, checkEps); }
    }

    void ComponentSearchFlowLoadWeight_smart::prepareForSearch_smart_intersect(unsigned query, std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                     std::vector<unsigned> &checkEps)
    {
        std::vector<float> weight = smart_index->getSearchWeight_q(query);
        std::set<int> needCalField_set;

        needCalField.clear();
        needIndeces.clear();
        for (int i = 0; i < weight.size(); i++)
        {
            if (weight[i])
            {
                needCalField.emplace_back(i);
                needCalField_set.insert(i);
            }
        }

        // ### --- intersect ---
        {
            for (int ind = 0; ind < smart_index->getGroupNum(); ind++)
            {
                auto &fElements = smart_index->getGroupList()[ind];
                bool skip = true;
                for (auto f : fElements)
                {
                    if (needCalField_set.find(f) != needCalField_set.end())
                    {
                        skip = false;
                        break;
                    }
                }
                if (!skip)
                {
                    needIndeces.emplace_back(ind);
                }
            }
        }
    }

    void ComponentSearchFlowLoadWeight_smart::prepareForSearch_smart_SetNeedCalField(unsigned query, std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                     std::vector<unsigned> &checkEps)
    {
        std::vector<float> weight = smart_index->getSearchWeight_q(query);
        needCalField.clear();
        for (int i = 0; i < weight.size(); i++)
        {
            if (weight[i])
            {
                needCalField.emplace_back(i);
            }
        }
    }

    void ComponentSearchFlowLoadWeight_smart::prepareForSearch_smart_allIndex(unsigned query, std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                                std::vector<unsigned> &checkEps)
    {
        std::vector<float> weight = smart_index->getSearchWeight_q(query);
        unsigned group = 0;

        needCalField.clear();
        needIndeces.clear();
        for (int i = 0; i < weight.size(); i++)
        {
            if (weight[i])
            {
                needCalField.emplace_back(i);
            }
        }
        // ## all index
        needIndeces.resize(smart_index->getGroupNum());
        for (int ind = 0; ind < smart_index->getGroupNum(); ind++)
        {
            needIndeces[ind] = ind;
        }

        // ## show needIndeces
        if (query == 0)
        {
            std::cout << "- needIndeces: ";
            for (auto ind : needIndeces)
            {
                std::cout << ind << ", ";
            }
            std::cout << std::endl;
            std::cout << "- needIndeces end" << std::endl;
        }
    }

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnce_smart_FallbackIntersect(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                         float &recall, float &latency, float &hop, float &distCount, TYPE dist_type)
    {
        // ## 0. preliminary: parameters & summary
        std::cout << "__weight_type == LOAD_WEIGHT__" << std::endl;
        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        res_list.resize(smart_index->getQueryLen());
        dist_res_list.resize(smart_index->getQueryLen());

        assert(L >= K);
        smart_index->getParam().set<unsigned>("L_search", L);

        std::cout << "### CHECK:K : " << K << std::endl;
        std::cout << "### CHECK:SEARCH_L : " << L << std::endl;
        std::cout << "smart_index->getBaseLen(): " << smart_index->getBaseLen() << std::endl;
        std::cout << "smart_index->getQueryLen(): " << smart_index->getQueryLen() << std::endl;

        auto s1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = s1 - s1, diff2 = s1 - s1, neg_time = s1 - s1;
        std::vector<int> needIndeces, needCalField, tmp_needIndeces;
        std::vector<unsigned> &checkEps = smart_index->each_ep_;

        // 如果构建时候定下的rela_thresh和查询时候想用的query_rela_thresh不一样，则根据bash_log中信息，选定一下使用哪个group: 如此书写仅为了方便WAGS的parameter-test
        if (smart_index->getSearchRelaSimThresh() != smart_index->getRelaThresh()) {
            std::cout << "❗️ ❗️ [ Warning ]: This are only allowed for doing prarmeterst test of c in WAGS， by pretending that the activation table we prepared is based on the current parameter values c = " << smart_index->getSearchRelaSimThresh() << ".❗️ ❗️ " << std::endl;
            std::cout << "__ [ NeedIndeces ] are get according to log information in (FlowInner_WeightOnce_smart_FallbackIntersect) __" << std::endl;

            std::string bash_path = "../include/python_file/bash.log_now";
            smart_index->preSetNeedIndeces_FallbackIntersect(smart_index->getSearchWeight_q(0), bash_path); 
            needIndeces = smart_index->getNeedIndeces();


            // ### 1. search
            // #pragma omp parallel for
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::vector<MultiIndex::Neighbor> pool;
                boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
                prepareForSearch_smart_SetNeedCalField(i, needCalField, needIndeces, checkEps); 

                auto s2 = std::chrono::high_resolution_clock::now();
                a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
                auto e2 = std::chrono::high_resolution_clock::now();
                diff1 += e2 - s2;

                // -- For Check & DEBUG --
                if (i == 0) {
                    std::cout << "--- Chosen 'Related Edge Group' for query == 0 ---------------------------------" << std::endl;
                    for (auto g : needIndeces) {
                        std::cout << g << ", ";
                    }
                    std::cout << "\n---------------------------------------------------------------------------------------------------------------------------\n\n" << std::endl;
                }
                
                s2 = std::chrono::high_resolution_clock::now();
                b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
                e2 = std::chrono::high_resolution_clock::now();
                diff2 += e2 - s2;
            }
        }
        else {
            std::cout << "__ [ NeedIndeces ] are set according to index (AGS-FallbackIntersect) __" << std::endl;
            // ### 1. search
            // #pragma omp parallel for
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::vector<MultiIndex::Neighbor> pool;
                boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
                prepareForSearch_smart_SavedAGS_FallbackIntersect(i, needCalField, needIndeces, checkEps); // 2026.03.25: 尝试savedAGS + Fallback-Intersect

                auto s2 = std::chrono::high_resolution_clock::now();
                a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
                auto e2 = std::chrono::high_resolution_clock::now();
                diff1 += e2 - s2;

                // -- For Check & DEBUG --
                if (i == 0) {
                    std::cout << "--- Chosen 'Related Edge Group' for query == 0 ---------------------------------" << std::endl;
                    for (auto g : needIndeces) {
                        std::cout << g << ", ";
                    }
                    std::cout << "\n---------------------------------------------------------------------------------------------------------------------------\n\n" << std::endl;
                }
                
                s2 = std::chrono::high_resolution_clock::now();
                b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
                e2 = std::chrono::high_resolution_clock::now();
                diff2 += e2 - s2;
            }
        }
        
        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e1 - s1;
        std::cout << "search time: " << diff.count() << "\n";
        std::cout << "SearchEntryInner time: " << diff1.count() << "\n";
        std::cout << "RouteInner_dist time: " << diff2.count() << "\n";
        // std::cout << "neg_time time: " << neg_time.count() << "\n";

        std::cout << "----[Average (per query) ---------------" << std::endl;
        std::cout << "average search time: " << (diff.count() / 100.0) << "\n";
        std::cout << "average SearchEntryInner time: " << (diff1.count() / 100.0) << "\n";
        std::cout << "average RouteInner_dist time: " << (diff2.count() / 100.0) << "\n";
        // std::cout << "average neg_time time: " << (neg_time.count() / 100.0) << "\n";
        std::cout << "-------------------" << std::endl;

        std::cout << "HopCount: " << smart_index->getHopCount() << std::endl;
        std::cout << "DistCount: " << smart_index->getDistCount() << std::endl;
        hop = smart_index->getHopCount();
        distCount = smart_index->getDistCount();
        smart_index->resetDistCount();
        smart_index->resetHopCount();


        // ### 2. 计算recall
        {
            if (dist_res_list[0].size() == 0)
            {
                for (unsigned i = 0; i < res_list.size(); i++)
                {
                    std::vector<int> needIndeces, needCalField;
                    std::vector<unsigned> checkEps;
                    prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                    for (unsigned j = 0; j < res_list[i].size(); j++)
                    {
                        float search_k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                                           smart_index->getBaseDataList(), res_list[i][j],
                                                                                           smart_index->getBaseDimList(), needCalField,
                                                                                           smart_index->getSearchWeight_q(i), dist_type);
                        dist_res_list[i].emplace_back(search_k_dist);
                    }
                }
            }
        }

        int cnt = 0;
        {
            // --- recall ---
            std::cout << "___ Recall: ___" << std::endl;
            for (unsigned i = 0; i < smart_index->getGroundLen(); i++) {
                if (dist_res_list[i].size() == 0) continue;
                std::vector<int> needIndeces, needCalField;
                std::vector<unsigned> checkEps;
                prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                float k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                for (unsigned j = 0; j < dist_res_list[i].size(); j++) {
                    dist_res_list[i][j] = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), res_list[i][j],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                }

                if (i < 2) {
                    std::cout << "-----------------------" << std::endl;
                    std::cout << "smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1]: " << smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1] << std::endl;
                    std::cout << "query-weight: ";
                    for (auto w : smart_index->getSearchWeight_q(i)) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needCalField: ";
                    for (auto w : needCalField) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;
                    
                    std::cout << "needIndeces: ";
                    for (auto w : needIndeces) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "k_dist: " << k_dist << std::endl;
                    std::cout << "-----------------------" << std::endl;
                }
                int j = K-1;
                for (; j >= 0; j--) {
                    if (static_cast<float>(dist_res_list[i][j]) > static_cast<float>(k_dist)) { cnt++; }
                    else {break;}
                }
            }
        }

        // ## 3. 展示ground-truth，以及search 结果
        std::cout << "--- For Check ---" << std::endl;
        if (L == 2000)
        {
            std::cout << "-- Check Result for L = " << L << "-------------------------------" << std::endl;
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::cout << "\n- For query " << i << std::endl;
                std::cout << "      -- Ground-truths: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;

                std::cout << "      -- Search Results: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                std::cout << "      -- with dist: ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << std::fixed << std::setprecision(16) << dist_res_list[i][j] << ", ";
                }
                std::cout << std::endl;
            }
            float f_check = 1.5;
            std::cout << std::fixed << std::setprecision(16) << "f_check: " << f_check << std::endl;
        }
        else
        {
            std::cout << "-- Ground-truths" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n-- Search Results" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                if (dist_res_list[0].size())
                {
                    std::cout << "dist: ";
                    for (unsigned j = 0; j < K; j++)
                    {
                        std::cout << dist_res_list[i][j] << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "---------------\n"
                  << std::endl;

        recall = 1 - (float)cnt / (smart_index->getGroundLen() * K);
        latency = diff.count();
        std::cout << K << " NN accuracy: " << recall << std::endl;
        std::cout << "search_time:" << std::to_string(latency) << std::endl;

    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnce_smart_FallbackIntersect_forDiff_wi(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                         float &recall, float &latency, float &hop, float &distCount, TYPE dist_type)
    {
        // ## 0. preliminary: parameters & summary
        std::cout << "__weight_type == LOAD_WEIGHT__" << std::endl;
        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        res_list.resize(smart_index->getQueryLen());
        dist_res_list.resize(smart_index->getQueryLen());

        assert(L >= K);
        smart_index->getParam().set<unsigned>("L_search", L);

        std::cout << "### CHECK:K : " << K << std::endl;
        std::cout << "### CHECK:SEARCH_L : " << L << std::endl;
        std::cout << "smart_index->getBaseLen(): " << smart_index->getBaseLen() << std::endl;
        std::cout << "smart_index->getQueryLen(): " << smart_index->getQueryLen() << std::endl;

        auto s1 = std::chrono::high_resolution_clock::now();
        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = s1 - s1, diff2 = s1 - s1, neg_time = s1 - s1, diff = s1 - s1;
        std::vector<int> needIndeces, needCalField, tmp_needIndeces;
        std::vector<unsigned> &checkEps = smart_index->each_ep_;

        // 如果构建时候定下的rela_thresh和查询时候想用的query_rela_thresh不一样，则根据bash_log中信息，选定一下使用哪个group: 如此书写仅为了方便WAGS的parameter-test
        if (smart_index->getSearchRelaSimThresh() != smart_index->getRelaThresh()) {
            std::cout << "❗️ ❗️ [ Warning ]: This are only allowed for doing prarmeterst test of c in WAGS， by pretending that the activation table we prepared is based on the current parameter values c = " << smart_index->getSearchRelaSimThresh() << ".❗️ ❗️ " << std::endl;
            std::cout << "__ [ NeedIndeces ] are get according to log information in (FlowInner_WeightOnce_smart_FallbackIntersect) __" << std::endl;

            std::string bash_path = "../include/python_file/bash.log_now";

            // ### 1. search
            // #pragma omp parallel for
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::vector<MultiIndex::Neighbor> pool;
                boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
                smart_index->preSetNeedIndeces_FallbackIntersect(smart_index->getSearchWeight_q(i), bash_path);
                s1 = std::chrono::high_resolution_clock::now();
                needIndeces = smart_index->getNeedIndeces();
                prepareForSearch_smart_SetNeedCalField(i, needCalField, needIndeces, checkEps); 

                auto s2 = std::chrono::high_resolution_clock::now();
                a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
                auto e2 = std::chrono::high_resolution_clock::now();
                diff1 += e2 - s2;

                // -- For Check & DEBUG --
                if (i == 0) {
                    std::cout << "--- Chosen 'Related Edge Group' for query == 0 ---------------------------------" << std::endl;
                    for (auto g : needIndeces) {
                        std::cout << g << ", ";
                    }
                    std::cout << "\n---------------------------------------------------------------------------------------------------------------------------\n\n" << std::endl;
                }
                
                s2 = std::chrono::high_resolution_clock::now();
                b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
                e2 = std::chrono::high_resolution_clock::now();
                diff2 += e2 - s2;

                e1 = std::chrono::high_resolution_clock::now();
                diff += e1-s1;
            }
        }
        else {
            std::cout << "__ [ NeedIndeces ] are set according to index (AGS-FallbackIntersect) __" << std::endl;
            s1 = std::chrono::high_resolution_clock::now();
            // ### 1. search
            // #pragma omp parallel for
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::vector<MultiIndex::Neighbor> pool;
                boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
                prepareForSearch_smart_SavedAGS_FallbackIntersect(i, needCalField, needIndeces, checkEps); // 2026.03.25: 尝试savedAGS + Fallback-Intersect

                auto s2 = std::chrono::high_resolution_clock::now();
                a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
                auto e2 = std::chrono::high_resolution_clock::now();
                diff1 += e2 - s2;

                // -- For Check & DEBUG --
                if (i == 0) {
                    std::cout << "--- Chosen 'Related Edge Group' for query == 0 ---------------------------------" << std::endl;
                    for (auto g : needIndeces) {
                        std::cout << g << ", ";
                    }
                    std::cout << "\n---------------------------------------------------------------------------------------------------------------------------\n\n" << std::endl;
                }
                
                s2 = std::chrono::high_resolution_clock::now();
                b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
                e2 = std::chrono::high_resolution_clock::now();
                diff2 += e2 - s2;
            }
            e1 = std::chrono::high_resolution_clock::now();
        }
        
        std::cout << "search time: " << diff.count() << "\n";
        std::cout << "SearchEntryInner time: " << diff1.count() << "\n";
        std::cout << "RouteInner_dist time: " << diff2.count() << "\n";
        // std::cout << "neg_time time: " << neg_time.count() << "\n";

        std::cout << "----[Average (per query) ---------------" << std::endl;
        std::cout << "average search time: " << (diff.count() / 100.0) << "\n";
        std::cout << "average SearchEntryInner time: " << (diff1.count() / 100.0) << "\n";
        std::cout << "average RouteInner_dist time: " << (diff2.count() / 100.0) << "\n";
        // std::cout << "average neg_time time: " << (neg_time.count() / 100.0) << "\n";
        std::cout << "-------------------" << std::endl;

        std::cout << "HopCount: " << smart_index->getHopCount() << std::endl;
        std::cout << "DistCount: " << smart_index->getDistCount() << std::endl;
        hop = smart_index->getHopCount();
        distCount = smart_index->getDistCount();
        smart_index->resetDistCount();
        smart_index->resetHopCount();


        // ### 2. 计算recall
        {
            if (dist_res_list[0].size() == 0)
            {
                for (unsigned i = 0; i < res_list.size(); i++)
                {
                    std::vector<int> needIndeces, needCalField;
                    std::vector<unsigned> checkEps;
                    prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                    for (unsigned j = 0; j < res_list[i].size(); j++)
                    {
                        float search_k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                                           smart_index->getBaseDataList(), res_list[i][j],
                                                                                           smart_index->getBaseDimList(), needCalField,
                                                                                           smart_index->getSearchWeight_q(i), dist_type);
                        dist_res_list[i].emplace_back(search_k_dist);
                    }
                }
            }
        }

        int cnt = 0;
        {
            // --- recall ---
            std::cout << "___ Recall: ___" << std::endl;
            for (unsigned i = 0; i < smart_index->getGroundLen(); i++) {
                if (dist_res_list[i].size() == 0) continue;
                std::vector<int> needIndeces, needCalField;
                std::vector<unsigned> checkEps;
                prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                float k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                for (unsigned j = 0; j < dist_res_list[i].size(); j++) {
                    dist_res_list[i][j] = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), res_list[i][j],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                }

                if (i < 2) {
                    std::cout << "-----------------------" << std::endl;
                    std::cout << "smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1]: " << smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1] << std::endl;
                    std::cout << "query-weight: ";
                    for (auto w : smart_index->getSearchWeight_q(i)) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needCalField: ";
                    for (auto w : needCalField) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;
                    
                    std::cout << "needIndeces: ";
                    for (auto w : needIndeces) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "k_dist: " << k_dist << std::endl;
                    std::cout << "-----------------------" << std::endl;
                }
                int j = K-1;
                for (; j >= 0; j--) {
                    if (static_cast<float>(dist_res_list[i][j]) > static_cast<float>(k_dist)) { cnt++; }
                    else {break;}
                }
            }
        }

        // ## 3. 展示ground-truth，以及search 结果
        std::cout << "--- For Check ---" << std::endl;
        if (L == 2000)
        {
            std::cout << "-- Check Result for L = " << L << "-------------------------------" << std::endl;
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::cout << "\n- For query " << i << std::endl;
                std::cout << "      -- Ground-truths: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;

                std::cout << "      -- Search Results: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                std::cout << "      -- with dist: ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << std::fixed << std::setprecision(16) << dist_res_list[i][j] << ", ";
                }
                std::cout << std::endl;
            }
            float f_check = 1.5;
            std::cout << std::fixed << std::setprecision(16) << "f_check: " << f_check << std::endl;
        }
        else
        {
            std::cout << "-- Ground-truths" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n-- Search Results" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                if (dist_res_list[0].size())
                {
                    std::cout << "dist: ";
                    for (unsigned j = 0; j < K; j++)
                    {
                        std::cout << dist_res_list[i][j] << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "---------------\n"
                  << std::endl;

        recall = 1 - (float)cnt / (smart_index->getGroundLen() * K);
        latency = diff.count();
        std::cout << K << " NN accuracy: " << recall << std::endl;
        std::cout << "search_time:" << std::to_string(latency) << std::endl;

    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnce_smart_intersect(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                         float &recall, float &latency, float &hop, float &distCount, TYPE dist_type)
    {
        std::cout << "__weight_type == LOAD_WEIGHT__" << std::endl; 

        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        res_list.resize(smart_index->getQueryLen());
        dist_res_list.resize(smart_index->getQueryLen());

        assert(L >= K);
        smart_index->getParam().set<unsigned>("L_search", L);

        std::cout << "### CHECK:K : " << K << std::endl;
        std::cout << "### CHECK:SEARCH_L : " << L << std::endl;
        std::cout << "smart_index->getBaseLen(): " << smart_index->getBaseLen() << std::endl;
        std::cout << "smart_index->getQueryLen(): " << smart_index->getQueryLen() << std::endl;
        /**/

        auto s1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff1 = s1 - s1, diff2 = s1 - s1, neg_time = s1 - s1;

        std::vector<int> needIndeces, needCalField, tmp_needIndeces;
        std::vector<unsigned> &checkEps = smart_index->each_ep_;

        // ### 1. search
        // #pragma omp parallel for
        for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
        {
            std::vector<MultiIndex::Neighbor> pool;
            boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
            prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);

            auto s2 = std::chrono::high_resolution_clock::now();
            a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
            auto e2 = std::chrono::high_resolution_clock::now();
            diff1 += e2 - s2;

            // -- For Check & DEBUG --
            if (i == 0) {
                std::cout << "--- Chosen 'Related Edge Group' for query == 0 ---------------------------------" << std::endl;
                for (auto g : needIndeces) {
                    std::cout << g << ", ";
                }
                std::cout << "\n---------------------------------------------------------------------------------------------------------------------------\n\n" << std::endl;
            }
            
            s2 = std::chrono::high_resolution_clock::now();
            b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
            e2 = std::chrono::high_resolution_clock::now();
            diff2 += e2 - s2;
        }

        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e1 - s1;
        std::cout << "search time: " << diff.count() << "\n";
        std::cout << "SearchEntryInner time: " << diff1.count() << "\n";
        std::cout << "RouteInner_dist time: " << diff2.count() << "\n";
        // std::cout << "neg_time time: " << neg_time.count() << "\n";

        std::cout << "----[Average (per query) ---------------" << std::endl;
        std::cout << "average search time: " << (diff.count() / 100.0) << "\n";
        std::cout << "average SearchEntryInner time: " << (diff1.count() / 100.0) << "\n";
        std::cout << "average RouteInner_dist time: " << (diff2.count() / 100.0) << "\n";
        // std::cout << "average neg_time time: " << (neg_time.count() / 100.0) << "\n";
        std::cout << "-------------------" << std::endl;

        std::cout << "HopCount: " << smart_index->getHopCount() << std::endl;
        std::cout << "DistCount: " << smart_index->getDistCount() << std::endl;
        hop = smart_index->getHopCount();
        distCount = smart_index->getDistCount();
        smart_index->resetDistCount();
        smart_index->resetHopCount();

        // ### 2. 计算recall
        {
            if (dist_res_list[0].size() == 0)
            {
                for (unsigned i = 0; i < res_list.size(); i++)
                {
                    std::vector<int> needIndeces, needCalField;
                    std::vector<unsigned> checkEps;
                    // prepareForSearch_smart_intersect(i, needIndeces, needCalField, checkEps);
                    prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                    for (unsigned j = 0; j < res_list[i].size(); j++)
                    {
                        float search_k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                                           smart_index->getBaseDataList(), res_list[i][j],
                                                                                           smart_index->getBaseDimList(), needCalField,
                                                                                           smart_index->getSearchWeight_q(i), dist_type);
                        dist_res_list[i].emplace_back(search_k_dist);
                    }
                }
            }
        }
        int cnt = 0;
        {
            // --- recall ---
            std::cout << "___ Recall: ___" << std::endl;
            for (unsigned i = 0; i < smart_index->getGroundLen(); i++) {
                if (dist_res_list[i].size() == 0) continue;
                std::vector<int> needIndeces, needCalField;
                std::vector<unsigned> checkEps;
                prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                float k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                for (unsigned j = 0; j < dist_res_list[i].size(); j++) {
                    dist_res_list[i][j] = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), res_list[i][j],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                }

                if (i < 2) {
                    std::cout << "-----------------------" << std::endl;
                    std::cout << "smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1]: " << smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1] << std::endl;
                    std::cout << "query-weight: ";
                    for (auto w : smart_index->getSearchWeight_q(i)) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needCalField: ";
                    for (auto w : needCalField) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needIndeces: ";
                    for (auto w : needIndeces) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "k_dist: " << k_dist << std::endl;
                    std::cout << "-----------------------" << std::endl;
                }
                int j = K-1;
                for (; j >= 0; j--) {
                    if (static_cast<float>(dist_res_list[i][j]) > static_cast<float>(k_dist)) { cnt++; }
                    else {break;}
                }
            }
        }

        // ## 3. 展示ground-truth，以及search 结果
        std::cout << "--- For Check ---" << std::endl;
        if (L == 2000)
        {
            std::cout << "-- Check Result for L = " << L << "-------------------------------" << std::endl;
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::cout << "\n- For query " << i << std::endl;
                std::cout << "      -- Ground-truths: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;

                std::cout << "      -- Search Results: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                std::cout << "      -- with dist: ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << std::fixed << std::setprecision(16) << dist_res_list[i][j] << ", ";
                }
                std::cout << std::endl;
            }
            float f_check = 1.5;
            std::cout << std::fixed << std::setprecision(16) << "f_check: " << f_check << std::endl;
        }
        else
        {
            std::cout << "-- Ground-truths" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n-- Search Results" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                if (dist_res_list[0].size())
                {
                    std::cout << "dist: ";
                    for (unsigned j = 0; j < K; j++)
                    {
                        std::cout << dist_res_list[i][j] << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "---------------\n"
                  << std::endl;

        recall = 1 - (float)cnt / (smart_index->getGroundLen() * K);
        latency = diff.count();
        std::cout << K << " NN accuracy: " << recall << std::endl;
        std::cout << "search_time:" << std::to_string(latency) << std::endl;
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnce_smart_exactRepre(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                    float &recall, float &latency, float &hop, float &distCount, TYPE dist_type)
    {
        std::cout << "__weight_type == LOAD_WEIGHT__" << std::endl;

        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        res_list.resize(smart_index->getQueryLen());
        dist_res_list.resize(smart_index->getQueryLen());

        assert(L >= K);
        smart_index->getParam().set<unsigned>("L_search", L);

        std::cout << "### CHECK:K : " << K << std::endl;
        std::cout << "### CHECK:SEARCH_L : " << L << std::endl;
        std::cout << "smart_index->getBaseLen(): " << smart_index->getBaseLen() << std::endl;
        std::cout << "smart_index->getQueryLen(): " << smart_index->getQueryLen() << std::endl;

        auto s1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = s1 - s1, diff2 = s1 - s1, neg_time = s1 - s1;

        // ### 1. search
        // #pragma omp parallel for
        for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
        {
            std::vector<int> needIndeces, needCalField;
            std::vector<unsigned> checkEps;
            std::vector<MultiIndex::Neighbor> pool;
            boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
            prepareForSearch_smart_exactRepre(i, needCalField, needIndeces, checkEps);

            auto s2 = std::chrono::high_resolution_clock::now();
            a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
            auto e2 = std::chrono::high_resolution_clock::now();
            diff1 += e2 - s2;

            s2 = std::chrono::high_resolution_clock::now();
            b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);
            e2 = std::chrono::high_resolution_clock::now();
            diff2 += e2 - s2;
        }

        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e1 - s1;
        std::cout << "search time: " << diff.count() << "\n";
        std::cout << "SearchEntryInner time: " << diff1.count() << "\n";
        std::cout << "RouteInner_dist time: " << diff2.count() << "\n";
        // std::cout << "neg_time time: " << neg_time.count() << "\n";

        std::cout << "----[Average (per query) ---------------" << std::endl;
        std::cout << "average search time: " << (diff.count() / 100.0) << "\n";
        std::cout << "average SearchEntryInner time: " << (diff1.count() / 100.0) << "\n";
        std::cout << "average RouteInner_dist time: " << (diff2.count() / 100.0) << "\n";
        // std::cout << "average neg_time time: " << (neg_time.count() / 100.0) << "\n";
        std::cout << "-------------------" << std::endl;

        std::cout << "HopCount: " << smart_index->getHopCount() << std::endl;
        std::cout << "DistCount: " << smart_index->getDistCount() << std::endl;
        hop = smart_index->getHopCount();
        distCount = smart_index->getDistCount();
        smart_index->resetDistCount();
        smart_index->resetHopCount();

        // ### 2. 计算recall
        {
            if (dist_res_list[0].size() == 0)
            {
                for (unsigned i = 0; i < res_list.size(); i++)
                {
                    std::vector<int> needIndeces, needCalField;
                    std::vector<unsigned> checkEps;
                    // prepareForSearch_smart_intersect(i, needIndeces, needCalField, checkEps);
                    prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                    for (unsigned j = 0; j < res_list[i].size(); j++)
                    {
                        float search_k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                                           smart_index->getBaseDataList(), res_list[i][j],
                                                                                           smart_index->getBaseDimList(), needCalField,
                                                                                           smart_index->getSearchWeight_q(i), dist_type);
                        dist_res_list[i].emplace_back(search_k_dist);
                    }
                }
            }
        }
        int cnt = 0;
        {
            // --- recall ---
            std::cout << "___ Recall: ___" << std::endl;
            for (unsigned i = 0; i < smart_index->getGroundLen(); i++) {
                if (dist_res_list[i].size() == 0) continue;
                std::vector<int> needIndeces, needCalField;
                std::vector<unsigned> checkEps;
                prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                float k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                for (unsigned j = 0; j < dist_res_list[i].size(); j++) {
                    dist_res_list[i][j] = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), res_list[i][j],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                }

                if (i < 2) {
                    std::cout << "-----------------------" << std::endl;
                    std::cout << "smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1]: " << smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1] << std::endl;
                    std::cout << "query-weight: ";
                    for (auto w : smart_index->getSearchWeight_q(i)) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needCalField: ";
                    for (auto w : needCalField) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needIndeces: ";
                    for (auto w : needIndeces) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "k_dist: " << k_dist << std::endl;
                    std::cout << "-----------------------" << std::endl;
                }
                int j = K-1;
                for (; j >= 0; j--) {
                    if (static_cast<float>(dist_res_list[i][j]) > static_cast<float>(k_dist)) { cnt++; }
                    else {break;}
                }
            }
        }

        // ## 3. 展示ground-truth，以及search 结果
        std::cout << "--- For Check ---" << std::endl;
        if (L == 2000)
        {
            std::cout << "-- Check Result for L = " << L << "-------------------------------" << std::endl;
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::cout << "\n- For query " << i << std::endl;
                std::cout << "      -- Ground-truths: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;

                std::cout << "      -- Search Results: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                std::cout << "      -- with dist: ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << std::fixed << std::setprecision(16) << dist_res_list[i][j] << ", ";
                }
                std::cout << std::endl;
            }
            float f_check = 2.5;
            std::cout << std::fixed << std::setprecision(16) << "f_check: " << f_check << std::endl;
        }
        else
        {
            std::cout << "-- Ground-truths" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n-- Search Results" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                if (dist_res_list[0].size())
                {
                    std::cout << "dist: ";
                    for (unsigned j = 0; j < K; j++)
                    {
                        std::cout << dist_res_list[i][j] << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "---------------\n"
                  << std::endl;

        recall = 1 - (float)cnt / (smart_index->getGroundLen() * K);
        latency = diff.count();
        std::cout << K << " NN accuracy: " << recall << std::endl;
        std::cout << "search_time:" << std::to_string(latency) << std::endl;
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnce_smart_allIndex(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                         float &recall, float &latency, float &hop, float &distCount, TYPE dist_type)
    {
        std::cout << "__weight_type == LOAD_WEIGHT__" << std::endl;

        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        res_list.resize(smart_index->getQueryLen());
        dist_res_list.resize(smart_index->getQueryLen());

        assert(L >= K);
        smart_index->getParam().set<unsigned>("L_search", L);

        std::cout << "### CHECK:K : " << K << std::endl;
        std::cout << "### CHECK:SEARCH_L : " << L << std::endl;
        std::cout << "smart_index->getBaseLen(): " << smart_index->getBaseLen() << std::endl;
        std::cout << "smart_index->getQueryLen(): " << smart_index->getQueryLen() << std::endl;

        auto s1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = s1 - s1, diff2 = s1 - s1, neg_time = s1 - s1;

        // ### 1. search
        // #pragma omp parallel for
        for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
        {
            std::vector<int> needIndeces, needCalField;
            std::vector<unsigned> &checkEps = smart_index->each_ep_;
            std::vector<MultiIndex::Neighbor> pool;
            boost::dynamic_bitset<> flags{smart_index->getBaseLen(), 0};
            prepareForSearch_smart_allIndex(i, needCalField, needIndeces, checkEps);

            auto s2 = std::chrono::high_resolution_clock::now();
            a->SearchEntryInner_smart(i, pool, needCalField, needIndeces, checkEps, flags, dist_type);
            auto e2 = std::chrono::high_resolution_clock::now();
            diff1 += e2 - s2;

            s2 = std::chrono::high_resolution_clock::now();
            b->RouteInner_dist_smart(i, pool, res_list[i], dist_res_list[i], needCalField, needIndeces, flags, dist_type);

            e2 = std::chrono::high_resolution_clock::now();
            diff2 += e2 - s2;
        }

        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e1 - s1;
        std::cout << "search time: " << diff.count() << "\n";
        std::cout << "SearchEntryInner time: " << diff1.count() << "\n";
        std::cout << "RouteInner_dist time: " << diff2.count() << "\n";
        // std::cout << "neg_time time: " << neg_time.count() << "\n";

        std::cout << "----[Average (per query) ---------------" << std::endl;
        std::cout << "average search time: " << (diff.count() / 100.0) << "\n";
        std::cout << "average SearchEntryInner time: " << (diff1.count() / 100.0) << "\n";
        std::cout << "average RouteInner_dist time: " << (diff2.count() / 100.0) << "\n";
        // std::cout << "average neg_time time: " << (neg_time.count() / 100.0) << "\n";
        std::cout << "-------------------" << std::endl;

        std::cout << "HopCount: " << smart_index->getHopCount() << std::endl;
        std::cout << "DistCount: " << smart_index->getDistCount() << std::endl;
        hop = smart_index->getHopCount();
        distCount = smart_index->getDistCount();
        smart_index->resetDistCount();
        smart_index->resetHopCount();

        // ### 2. 计算recall
        {
            if (dist_res_list[0].size() == 0)
            {
                for (unsigned i = 0; i < res_list.size(); i++)
                {
                    std::vector<int> needIndeces, needCalField;
                    std::vector<unsigned> checkEps;
                    prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                    for (unsigned j = 0; j < res_list[i].size(); j++)
                    {
                        float search_k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                                           smart_index->getBaseDataList(), res_list[i][j],
                                                                                           smart_index->getBaseDimList(), needCalField,
                                                                                           smart_index->getSearchWeight_q(i), dist_type);
                        dist_res_list[i].emplace_back(search_k_dist);
                    }
                }
            }
        }
        int cnt = 0;
        {
            // --- recall ---
            std::cout << "___ Recall: ___" << std::endl;
            for (unsigned i = 0; i < smart_index->getGroundLen(); i++) {
                if (dist_res_list[i].size() == 0) continue;
                std::vector<int> needIndeces, needCalField;
                std::vector<unsigned> checkEps;
                prepareForSearch_smart_intersect(i, needCalField, needIndeces, checkEps);
                float k_dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                for (unsigned j = 0; j < dist_res_list[i].size(); j++) {
                    dist_res_list[i][j] = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), i,
                                                                            smart_index->getBaseDataList(), res_list[i][j],
                                                                            smart_index->getBaseDimList(), needCalField,
                                                                            smart_index->getSearchWeight_q(i), dist_type);
                }

                if (i < 2) {
                    std::cout << "-----------------------" << std::endl;
                    std::cout << "smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1]: " << smart_index->getGroundData()[i * smart_index->getGroundDim() + K-1] << std::endl;
                    std::cout << "query-weight: ";
                    for (auto w : smart_index->getSearchWeight_q(i)) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needCalField: ";
                    for (auto w : needCalField) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;

                    std::cout << "needIndeces: ";
                    for (auto w : needIndeces) {
                        std::cout << w << ", ";
                    }
                    std::cout << std::endl;
                    
                    std::cout << "k_dist: " << k_dist << std::endl;
                    std::cout << "-----------------------" << std::endl;
                }
                int j = K-1;
                for (; j >= 0; j--) {
                    if (static_cast<float>(dist_res_list[i][j]) > static_cast<float>(k_dist)) { cnt++; }
                    else {break;}
                }
            }
        }
        // ## 3. 展示ground-truth，以及search 结果
        std::cout << "--- For Check ---" << std::endl;
        if (L == 2000)
        {
            std::cout << "-- Check Result for L = " << L << "-------------------------------" << std::endl;
            for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
            {
                std::cout << "\n- For query " << i << std::endl;
                std::cout << "      -- Ground-truths: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;

                std::cout << "      -- Search Results: " << std::endl;
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                std::cout << "      -- with dist: ";
                for (unsigned j = 0; j < K; j++)
                {
                    // std::cout << std::fixed << std::setprecision(7) << dist_res_list[i][j] << ", ";
                    std::cout << std::fixed << std::setprecision(16) << dist_res_list[i][j] << ", ";
                }
                std::cout << std::endl;
            }
            float f_check = 1.5;
            std::cout << std::fixed << std::setprecision(16) << "f_check: " << f_check << std::endl;
        }
        else
        {
            std::cout << "-- Ground-truths" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << smart_index->getGroundData()[i * smart_index->getGroundDim() + j] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "\n-- Search Results" << std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                std::cout << "- For query " << i << "\n     ";
                for (unsigned j = 0; j < K; j++)
                {
                    std::cout << res_list[i][j] << ", ";
                }
                std::cout << std::endl;
                if (dist_res_list[0].size())
                {
                    std::cout << "dist: ";
                    for (unsigned j = 0; j < K; j++)
                    {
                        std::cout << dist_res_list[i][j] << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "---------------\n"
                  << std::endl;

        recall = 1 - (float)cnt / (smart_index->getGroundLen() * K);
        latency = diff.count();
        std::cout << K << " NN accuracy: " << recall << std::endl;
        std::cout << "search_time:" << std::to_string(latency) << std::endl;
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnceControL_smart_FallbackIntersect(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                                                TYPE dist_type)
    {
        unsigned L;
        for (unsigned ll = 0; ll < LRate.size(); ll++)
        {
            L = LRate[ll] * K;
            FlowInner_WeightOnce_smart_FallbackIntersect(K, L, a, b, recall_list[ll], latency_list[ll], hop_list[ll], distCount_list[ll], dist_type);
        }
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnceControL_smart_FallbackIntersect_forDiff_wi(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                                                TYPE dist_type)
    {
        unsigned L;
        for (unsigned ll = 0; ll < LRate.size(); ll++)
        {
            L = LRate[ll] * K;
            FlowInner_WeightOnce_smart_FallbackIntersect_forDiff_wi(K, L, a, b, recall_list[ll], latency_list[ll], hop_list[ll], distCount_list[ll], dist_type);
        }
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnceControL_smart_intersect(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                                                TYPE dist_type)
    {
        unsigned L;
        for (unsigned ll = 0; ll < LRate.size(); ll++)
        {
            L = LRate[ll] * K;
            FlowInner_WeightOnce_smart_intersect(K, L, a, b, recall_list[ll], latency_list[ll], hop_list[ll], distCount_list[ll], dist_type);
        }
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnceControL_smart_exactRepre(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                           std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                                                           std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                                                           TYPE dist_type)
    {
        unsigned L;
        for (unsigned ll = 0; ll < LRate.size(); ll++)
        {
            L = LRate[ll] * K;
            FlowInner_WeightOnce_smart_exactRepre(K, L, a, b, recall_list[ll], latency_list[ll], hop_list[ll], distCount_list[ll], dist_type);
        }
    };

    void ComponentSearchFlowLoadWeight_smart::FlowInner_WeightOnceControL_smart_allIndex(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b,
                                                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                                                TYPE dist_type)
    {
        unsigned L;
        for (unsigned ll = 0; ll < LRate.size(); ll++)
        {
            L = LRate[ll] * K;
            FlowInner_WeightOnce_smart_allIndex(K, L, a, b, recall_list[ll], latency_list[ll], hop_list[ll], distCount_list[ll], dist_type);
        }
    };
}
