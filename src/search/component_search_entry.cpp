#include "component.h"


// ======================================
// Search Entry Point
// ======================================

namespace xmt {

    void ComponentSearchEntryNone_Fusion::SearchEntryInner_multi(unsigned query, std::vector<MultiIndex::Neighbor> &pool,
                                                                 std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                 std::vector<unsigned> &checkEps, boost::dynamic_bitset<> &flags,
                                                                 TYPE dist_type) {};

    void ComponentSearchEntryCentroid_multi::SearchEntryInner_multi(unsigned int query, std::vector<MultiIndex::Neighbor> &pool,
                                                                    std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                    std::vector<unsigned> &checkEps, boost::dynamic_bitset<> &flags, TYPE dist_type)
    {
        const auto L = std::min(smart_index->getParam().get<unsigned>("L_search"), smart_index->getBaseLen());

        std::vector<unsigned> init_ids;
        std::mt19937 rng(777);
        int group = smart_index->getGroupNum() - 1;
        std::vector<float> weight = smart_index->getSearchWeight_q(query);
        unsigned ep;

        pool.clear();
        for (unsigned id : checkEps)
        {
            if (flags[id])
            {
                continue;
            }
            float dist = smart_index->getDist()->compare_multi_weight(smart_index->getQueryDataList(), query,
                                                                      smart_index->getBaseDataList(), id,
                                                                      smart_index->getBaseDimList(), needCalField,
                                                                      weight, dist_type);
            smart_index->addDistCount();
            pool.emplace_back(MultiIndex::Neighbor(id, dist, true));
            flags[id] = true;
        }
        std::sort(pool.begin(), pool.end());

        float place_holder_dist = MAXFLOAT;
        unsigned place_holder = pool[pool.size()-1].id;
        while (pool.size() < L) {
            pool.emplace_back(MultiIndex::Neighbor(place_holder, place_holder_dist, true));
        }

        pool.resize(L);
        pool.reserve(L + 1);
    };
}
