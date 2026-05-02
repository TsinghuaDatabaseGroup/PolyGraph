//
// Created by mengtong-x on 2024/06/23.
//

#include "component.h"


// ======================================
// Search Entry Point
// ======================================

namespace weavess {
    void ComponentSearchRouteGreedy_smart::RouteInner_dist_smart(unsigned int query, std::vector<SmartIndex::Neighbor> &pool,
                                                                 std::vector<unsigned int> &res, std::vector<float> &dist_res,
                                                                 std::vector<int> &needCalField, std::vector<int> &needIndeces,
                                                                 boost::dynamic_bitset<> &flags, TYPE dist_type)
    {
        const auto K = smart_index->getParam().get<unsigned>("K_search");
        auto L = smart_index->getParam().get<unsigned>("L_search");
        L = std::min(L, smart_index->getBaseLen());


        int k = 0;

        // ## 1. get weight
        std::vector<float> weight = smart_index->getSearchWeight_q(query);

        // ## 2. GREEDY SEARCH
        unsigned tmp_hop = 0;
        while (k < (int)L)
        {
            int nk = L;
            if (pool[k].flag)
            {
                pool[k].flag = false;
                unsigned n = pool[k].id;

                smart_index->addHopCount();
                tmp_hop++;
                for (auto g : needIndeces)
                {
                    for (int m = 0; m < smart_index->getLoadGraph(g)[n].size(); m++)
                    {
                        unsigned id = smart_index->getLoadGraph(g)[n][m];

                        if (flags[id])
                            continue;
                        flags[id] = 1;
                        float dist = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), query,
                                                                                  smart_index->getBaseDataList(), id,
                                                                                  smart_index->getBaseDimList(), needCalField,
                                                                                  weight, dist_type);
                        smart_index->addDistCount();
                        if (dist > pool[L - 1].distance)
                        {
                            continue;
                        } 
                        SmartIndex::Neighbor nn(id, dist, true);
                        int r = SmartIndex::InsertIntoPool(pool.data(), L, nn);
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
        res.resize(K);
        dist_res.resize(K);
        for (size_t i = 0; i < K; i++)
        {
            res[i] = pool[i].id;
            dist_res[i] = pool[i].distance;
        }
    };



    void ComponentSearchRouteHNSW_Fusion::RouteInner_dist_smart(unsigned query, std::vector<SmartIndex::Neighbor> &pool, std::vector<unsigned> &res, std::vector<float> &dist_res,
                                                                std::vector<int> &needCalField, std::vector<int> &needIndeces, boost::dynamic_bitset<> &flags, TYPE dist_type)
    {
        const auto K = smart_index->getParam().get<unsigned>("K_search");

        auto *visited_list = new SmartIndex::VisitedList(smart_index->getBaseLen());
        std::vector<float> weight = smart_index->getSearchWeight_q(query);

        SmartIndex::HnswNode *enterpoint = smart_index->enterpoint_;
        std::vector<std::pair<SmartIndex::HnswNode *, float>> ensure_k_path_; 
        SmartIndex::HnswNode *cur_node = enterpoint;

        float d = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), query,
                                                               smart_index->getBaseDataList(), cur_node->GetId(),
                                                               smart_index->getBaseDimList(), needCalField,
                                                               weight, TYPE::DIST_EUCLIDEAN);
        smart_index->addDistCount();
        float cur_dist = d;

        ensure_k_path_.clear();
        ensure_k_path_.emplace_back(cur_node, cur_dist);

        for (auto i = smart_index->max_level_; i >= 0; --i)
        {
            visited_list->Reset();
            unsigned visited_mark = visited_list->GetVisitMark();
            unsigned int *visited = visited_list->GetVisited();
            visited[cur_node->GetId()] = visited_mark;

            bool changed = true;
            while (changed)
            {
                changed = false;
                std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                const std::vector<SmartIndex::HnswNode *> &neighbors = cur_node->GetFriends(i);

                smart_index->addHopCount();
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                {
                    if (visited[(*iter)->GetId()] != visited_mark)
                    {
                        visited[(*iter)->GetId()] = visited_mark;
                        d = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), query,
                                                                         smart_index->getBaseDataList(), (*iter)->GetId(),
                                                                         smart_index->getBaseDimList(), needCalField,
                                                                         weight, TYPE::DIST_EUCLIDEAN);
                        smart_index->addDistCount();
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                            ensure_k_path_.emplace_back(cur_node, cur_dist);
                        }
                    }
                }
            }
        }

        std::priority_queue<SmartIndex::FurtherFirst> result;
        std::priority_queue<SmartIndex::CloserFirst> tmp;

        while (result.size() < K && !ensure_k_path_.empty())
        {
            cur_dist = ensure_k_path_.back().second;
            SearchAtLayer(query, ensure_k_path_.back().first, 0, weight, needCalField, visited_list, result, dist_type);
            ensure_k_path_.pop_back();
        }
        while (!result.empty())
        {
            tmp.push(SmartIndex::CloserFirst(result.top().GetNode(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(K);
        int pos = 0;
        while (!tmp.empty() && pos < K)
        {
            auto *top_node = tmp.top().GetNode();
            tmp.pop();
            res[pos] = top_node->GetId();
            pos++;
        }
        delete visited_list;
    }

    void ComponentSearchRouteHNSW_Fusion::SearchAtLayer(unsigned qnode, SmartIndex::HnswNode *enterpoint, int level,
                                                        std::vector<float> &weight, std::vector<int> &needCalField, SmartIndex::VisitedList *visited_list,
                                                        std::priority_queue<SmartIndex::FurtherFirst> &result, TYPE dist_type)
    {
        const auto L = std::min(smart_index->getParam().get<unsigned>("L_search"), smart_index->getBaseLen());

        std::priority_queue<SmartIndex::CloserFirst> candidates;
        float d = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), qnode,
                                                               smart_index->getBaseDataList(), enterpoint->GetId(),
                                                               smart_index->getBaseDimList(), needCalField,
                                                               weight, TYPE::DIST_EUCLIDEAN);
        smart_index->addDistCount();
        result.emplace(enterpoint, d);
        candidates.emplace(enterpoint, d);

        visited_list->Reset();
        visited_list->MarkAsVisited(enterpoint->GetId());

        while (!candidates.empty())
        {
            const SmartIndex::CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound)
                break;

            SmartIndex::HnswNode *candidate_node = candidate.GetNode();
            std::unique_lock<std::mutex> lock(candidate_node->GetAccessGuard());
            const std::vector<SmartIndex::HnswNode *> &neighbors = candidate_node->GetFriends(level);
            candidates.pop();
            smart_index->addHopCount();
            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    d = smart_index->getDist()->compare_smart_weight(smart_index->getQueryDataList(), qnode,
                                                                     smart_index->getBaseDataList(), id,
                                                                     smart_index->getBaseDimList(), needCalField,
                                                                     weight, TYPE::DIST_EUCLIDEAN);
                    smart_index->addDistCount();
                    if (result.size() < L || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > L)
                            result.pop();
                    }
                }
            }
        }
    }
}
