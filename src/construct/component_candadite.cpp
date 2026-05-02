//
// Created by mengtong-x on 2026/04/30.
//

#include "component.h"


// ======================================
// Candidate Neighbor Acquisition
// ======================================
namespace weavess {
    // ----------------------------------------------------------------------------------------------------
    // PolyGraph, Vamana-series
    // ----------------------------------------------------------------------------------------------------
    void ComponentCandidateAGS_smart::CandidateInner_smart_4group_STAR_allIndex(
        const unsigned query, const unsigned enter, int group, boost::dynamic_bitset<> flags,
        std::vector<SmartIndex::SimpleNeighbor> &result, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        auto L = smart_index->L;
        auto L_refine_whole = smart_index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids;
        std::vector<SmartIndex::Neighbor> retset;

        {
            if (enter != query)
            {
                init_ids.emplace_back(enter);
            }
            flags[enter] = true;
            for (int f = 0; f < smart_index->getGroupNum(); f++)
            {
                for (unsigned i = 0; i < smart_index->getOldGraph(f)[enter].size(); i++)
                {
                    unsigned id = smart_index->getOldGraph(f)[enter][i].id;
                    if (flags[id] || id == query)
                        continue;
                    if (id >= smart_index->getBaseLen())
                    {
                        std::cout << "-- EXCEED Baselen, of [enter" << enter << "][" << i << "]: " << init_ids[i] << std::endl;
                        while (id >= smart_index->getBaseLen())
                        {
                            id = rand() % smart_index->getBaseLen();
                        }
                        exit(-1);
                    }
                    flags[id] = true;
                    init_ids.emplace_back(id);
                }
            }
        }

        while (init_ids.size() < L_refine_whole)
        {
            unsigned id = rand() % smart_index->getBaseLen();
            if (flags[id] || id == query)
                continue;
            init_ids.emplace_back(id);
            flags[id] = true;
        }
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                      smart_index->getBaseDataList(), id,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);

            retset.emplace_back(SmartIndex::Neighbor(id, dist, true));
        }
        std::sort(retset.begin(), retset.end());
        retset.resize(L + 1);
        smart_index->i++;

        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                result.emplace_back(SmartIndex::SimpleNeighbor(retset[k].id, retset[k].distance));
                for (int f = 0; f < smart_index->getGroupNum(); f++)
                {
                    auto graph_q = smart_index->getOldGraph(f)[n];
                    for (unsigned m = 0; m < graph_q.size(); ++m)
                    {
                        unsigned id = graph_q[m].id;
                        if (id >= smart_index->getBaseLen())
                        {
                            std::cout << " -- EXCEED Baselen, of [" << n << "][" << m << "]: " << id << std::endl;
                            exit(-1);
                        }

                        if (flags[id] || id == query)
                            continue;
                        flags[id] = true;

                        float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                                  smart_index->getBaseDataList(), id,
                                                                                  smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                                  smart_index->getGroupRepreList()[group], dist_type);

                        SmartIndex::Neighbor nn(id, dist, true);
                        
                        if (dist >= retset[L - 1].distance)
                            continue; 

                        int r = SmartIndex::InsertIntoPool(retset.data(), L, nn);
                        if (L + 1 < retset.size())
                            ++L;
                        if (r < nk)
                            nk = r;
                    }
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

        
        {
            auto graph_q = smart_index->getOldGraph(group)[query];
            for (unsigned pos = 0; pos < graph_q.size(); pos++)
            {
                unsigned id = graph_q[pos].id;
                if (flags[id] || id == query)
                    continue;
                flags[id] = true;

                float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                          smart_index->getBaseDataList(), id,
                                                                          smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                          smart_index->getGroupRepreList()[group], dist_type);

                result.push_back(SmartIndex::SimpleNeighbor(id, dist));
            }
        }
        std::vector<SmartIndex::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }




    // ----------------------------------------------------------------------------------------------------
    // Oracle
    // ----------------------------------------------------------------------------------------------------
    void ComponentCandidateAGS_smart::CandidateInner_smart_4group_STAR(
        const unsigned query, const unsigned enter, int group, boost::dynamic_bitset<> flags,
        std::vector<SmartIndex::SimpleNeighbor> &result, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        auto L = smart_index->L;
        auto L_refine_whole = smart_index->getParam().get<unsigned>("L_refine");

        std::vector<unsigned> init_ids;
        std::vector<SmartIndex::Neighbor> retset;
        std::vector<int> relaGroups = smart_index->getRelaCheck(group);

        {
            if (enter != query)
            {
                init_ids.emplace_back(enter);
            }
            flags[enter] = true;
            for (auto f : relaGroups)
            {
                for (unsigned i = 0; i < smart_index->getOldGraph(f)[enter].size(); i++)
                {
                    unsigned id = smart_index->getOldGraph(f)[enter][i].id;
                    if (flags[id] || id == query)
                        continue;
                    if (id >= smart_index->getBaseLen())
                    {
                        std::cout << "-- EXCEED Baselen, of [enter" << enter << "][" << i << "]: " << init_ids[i] << std::endl;
                        while (id >= smart_index->getBaseLen())
                        {
                            id = rand() % smart_index->getBaseLen();
                        }
                        exit(-1);
                    }
                    flags[id] = true;
                    init_ids.emplace_back(id);
                }
            }
        }

        while (init_ids.size() < L_refine_whole)
        {
            unsigned id = rand() % smart_index->getBaseLen();
            if (flags[id] || id == query)
                continue;
            init_ids.emplace_back(id);
            flags[id] = true;
        }
        
        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                      smart_index->getBaseDataList(), id,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);

            retset.emplace_back(SmartIndex::Neighbor(id, dist, true));
        }
        std::sort(retset.begin(), retset.end());
        retset.resize(L + 1);
        smart_index->i++;
        
        int k = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (retset[k].flag)
            {
                retset[k].flag = false;
                unsigned n = retset[k].id;
                result.emplace_back(SmartIndex::SimpleNeighbor(retset[k].id, retset[k].distance));
                for (auto f : relaGroups)
                {
                    auto graph_q = smart_index->getOldGraph(f)[n];
                    for (unsigned m = 0; m < graph_q.size(); ++m)
                    {
                        unsigned id = graph_q[m].id;
                        if (id >= smart_index->getBaseLen())
                        {
                            std::cout << " -- EXCEED Baselen, of [" << n << "][" << m << "]: " << id << std::endl;
                            exit(-1);
                        }

                        if (flags[id] || id == query)
                            continue;
                        flags[id] = true;

                        float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                                  smart_index->getBaseDataList(), id,
                                                                                  smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                                  smart_index->getGroupRepreList()[group], dist_type);
                        SmartIndex::Neighbor nn(id, dist, true);

                        if (dist >= retset[L - 1].distance)
                            continue; 

                        int r = SmartIndex::InsertIntoPool(retset.data(), L, nn);

                        if (L + 1 < retset.size())
                            ++L;
                        if (r < nk)
                            nk = r;
                    }
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }

        {
            auto graph_q = smart_index->getOldGraph(group)[query];
            for (unsigned pos = 0; pos < graph_q.size(); pos++)
            {
                unsigned id = graph_q[pos].id;
                if (flags[id] || id == query)
                    continue;
                flags[id] = true;

                float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                          smart_index->getBaseDataList(), id,
                                                                          smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                          smart_index->getGroupRepreList()[group], dist_type);

                result.push_back(SmartIndex::SimpleNeighbor(id, dist));
            }
        }
        std::vector<SmartIndex::Neighbor>().swap(retset);
        std::vector<unsigned>().swap(init_ids);
    }
}