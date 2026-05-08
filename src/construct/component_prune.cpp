#include "component.h"


// ======================================
// Neighborhood Construction
//              ----> Neighbor Selection
// ======================================
namespace xmt {

    // ----------------------------------------------------------------------------------------------------
    // PolyGraph
    // ----------------------------------------------------------------------------------------------------
    void ComponentPrunePolyGraph_multi::PruneInner_multi_4group_withOthers(unsigned query, int group, std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        std::vector<MultiIndex::SimpleNeighbor> picked;
        std::vector<MultiIndex::SimpleNeighbor> other_selected;

        std::set<unsigned> had_ON_before, had_ON_this;

        std::vector<int> rela_check = smart_index->getRelaCheck(group);

        // ## --- 0. 将related_group 中的ON都放入到 had_ON_before 中去
        for (auto g : rela_check)
        {
            if (g <= group)
            {
                continue;
            }
            for (unsigned i = 0; i < smart_index->getFinalGraph(g)[query].size(); i++)
            {
                float id = smart_index->getFinalGraph(g)[query][i].id;
                if (had_ON_before.find(id) == had_ON_before.end())
                {
                    had_ON_before.emplace(id);
                    float dist = smart_index->getDist()->compare_multi_weight(smart_index->getBaseDataList(), query,
                                                                              smart_index->getBaseDataList(), id,
                                                                              smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                              smart_index->getGroupRepreList()[group], dist_type);
                    pool.emplace_back(MultiIndex::SimpleNeighbor(id, dist));
                }
            }
        }
        std::sort(pool.begin(), pool.end());

        // ## 1. 确定最终prune后的元素：picked
        unsigned this_id;
        for (int i = 0; i < pool.size(); i++)
        {
            this_id = pool[i].id;
            if (this_id == query)
                continue;
            // -- 1.1 如果 had_ON_before 中已经有了，直接记入other_selected
            if (had_ON_before.find(this_id) != had_ON_before.end())
            {
                other_selected.emplace_back(pool[i]);
                continue;
            }
            // -- 1.2 如果 had_ON_this 中已经有了，说明出现重复，这个可以跳过了
            if (had_ON_this.find(this_id) != had_ON_this.end())
            {
                continue;
            }

            bool skip = false;
            float cur_dist = pool[i].distance;
            {
                // ---- 与自己field中picked对比
                for (size_t j = 0; j < picked.size(); j++)
                {
                    float dist = smart_index->getDist()->compare_multi_weight(smart_index->getBaseDataList(), picked[j].id,
                                                                              smart_index->getBaseDataList(), this_id,
                                                                              smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                              smart_index->getGroupRepreList()[group], dist_type);
                    if (smart_index->alpha * dist <= cur_dist) 
                    {
                        skip = true;
                        break;
                    }
                }

                // --- 与其他group中比自己dist小的区域对比
                if (!skip)
                {
                    for (size_t j = 0; j < other_selected.size(); j++)
                    {
                        float dist = smart_index->getDist()->compare_multi_weight(smart_index->getBaseDataList(), other_selected[j].id,
                                                                                  smart_index->getBaseDataList(), this_id,
                                                                                  smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                                  smart_index->getGroupRepreList()[group], dist_type);
                        if (smart_index->alpha * dist <= cur_dist)
                        {
                            skip = true;
                            break;
                        }
                    }
                }
            }

            if (!skip)
            {
                picked.push_back(pool[i]);
                had_ON_this.emplace(this_id);
            }

            if (picked.size() == smart_index->R)
            {
                break;
            }
        }

        // ## 2. 根据picked对应修改group-th graph
        MultiIndex::LockGuard guard(locks[query]);
        auto &graph_q = smart_index->getFinalGraph(group)[query];
        {
            graph_q.resize(std::min(smart_index->R, (unsigned)picked.size()));
            for (size_t t = 0; t < std::min(smart_index->R, (unsigned)picked.size()); t++)
            {
                graph_q[t].id = picked[t].id;
                graph_q[t].distance = picked[t].distance;
            }

            std::vector<MultiIndex::SimpleNeighbor>().swap(picked);
        }
    }



    // ----------------------------------------------------------------------------------------------------
    // HNSW_Fusion
    // ----------------------------------------------------------------------------------------------------
    void ComponentPruneHeuristic_multi::PruneInner_Fusion(unsigned query, unsigned int range, // boost::dynamic_bitset<> flags,
                                                          std::vector<MultiIndex::SimpleNeighbor> &pool, MultiIndex::SimpleNeighbor *cut_graph_,
                                                          TYPE dist_type)
    {
        std::vector<int> need_calcu;
        for (int i = 0; i < smart_index->getFieldNum(); i++)
        {
            need_calcu.emplace_back(i);
        }

        std::vector<MultiIndex::SimpleNeighbor> picked;
        if (pool.size() > range)
        {
            for (int i = 0; i < pool.size(); i++)
            {
                bool skip = false;
                float cur_dist = pool[i].distance;
                for (size_t j = 0; j < picked.size(); j++)
                {
                    float dist = smart_index->getDist()->compare_multi(smart_index->getBaseDataList(), picked[j].id,
                                                                       smart_index->getBaseDataList(), pool[i].id,
                                                                       smart_index->getBaseDimList(), need_calcu,
                                                                       TYPE::DIST_EUCLIDEAN);
                        if(dist <= cur_dist) { 
                        skip = true;
                        break;
                    }
                }

                if (!skip)
                {
                    picked.push_back(pool[i]);
                }

                if (picked.size() == range)
                    break;
            }
        }
        else
        {
            for (int i = 0; i < pool.size(); i++)
            {
                picked.push_back(pool[i]);
            }
        }
        MultiIndex::SimpleNeighbor *des_pool = cut_graph_ + (size_t)query * (size_t)range;
        for (size_t t = 0; t < picked.size(); t++)
        {
            des_pool[t].id = picked[t].id;
            des_pool[t].distance = picked[t].distance;
        }

        if (picked.size() < range)
        {
            des_pool[picked.size()].distance = -1;
        }

        std::vector<MultiIndex::SimpleNeighbor>().swap(picked);
    }






    // ----------------------------------------------------------------------------------------------------
    // Vamana-series
    // ----------------------------------------------------------------------------------------------------
    void ComponentPruneVamana_multi::PruneInner_multi_4group_withOthers(unsigned query, int group, std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        std::vector<MultiIndex::SimpleNeighbor> picked;
        std::set<unsigned> had_ON_this;
        std::sort(pool.begin(), pool.end());

        // ## 1. 确定最终prune后的元素：picked
        unsigned this_id;
        for (int i = 0; i < pool.size(); i++)
        {
            this_id = pool[i].id;
            if (this_id == query)
                continue;
            // -- had_ON_this 中已经有了，说明出现重复，这个可以跳过了
            if (had_ON_this.find(this_id) != had_ON_this.end())
            {
                continue;
            }

            bool skip = false;
            float cur_dist = pool[i].distance;
            {
                // ---- 与自己field中picked对比
                for (size_t j = 0; j < picked.size(); j++)
                {
                    float dist = smart_index->getDist()->compare_multi_weight(smart_index->getBaseDataList(), picked[j].id,
                                                                              smart_index->getBaseDataList(), this_id,
                                                                              smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                              smart_index->getGroupRepreList()[group], dist_type);
                    if (smart_index->alpha * dist <= cur_dist) 
                    {
                        skip = true;
                        break;
                    }
                }
            }

            if (!skip)
            {
                picked.push_back(pool[i]);
                had_ON_this.emplace(this_id);
            }

            if (picked.size() == smart_index->R)
            {
                break;
            }
        }

        // ## 2. 根据picked对应修改group-th graph
        MultiIndex::LockGuard guard(locks[query]);
        auto &graph_q = smart_index->getFinalGraph(group)[query];
        {
            graph_q.resize(std::min(smart_index->R, (unsigned)picked.size()));
            for (size_t t = 0; t < std::min(smart_index->R, (unsigned)picked.size()); t++)
            {
                graph_q[t].id = picked[t].id;
                graph_q[t].distance = picked[t].distance;
            }

            std::vector<MultiIndex::SimpleNeighbor>().swap(picked);
        }
    }
}