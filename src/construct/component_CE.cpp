//
// Created by mengtong-x on 2026/04/30.
//

#include "component.h"


// ======================================
// graph connectivity enforcer
// ======================================
namespace weavess {
    void ComponentRelaConnectEnforcer::ConnectEnforcerInner_smart(TYPE dist_type)
    {
        std::set<unsigned> isolated_ids;
        for (int g = 0; g < smart_index->getGroupNum(); g++)
        {
            auto s_c = std::chrono::high_resolution_clock::now();

            if (FindIsolate(smart_index->each_ep_[g], smart_index->getRelaCheck(g), isolated_ids))
            {
                ConnectAndSpread(g, smart_index->getRelaCheck(g), isolated_ids);
            }

            auto e_c = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> load_info_time = e_c - s_c;
            std::cout << "^^^^^^ connect() time for group-" << g << " is " << load_info_time.count() << " ^^^^^^" << std::endl;
        }
    }

    bool ComponentRelaConnectEnforcer::FindIsolate(unsigned ep, std::vector<int> relaGroup, std::set<unsigned> &isolate_ids)
    {
        std::cout << "____ FIND_ISOLATE_ID: START ____" << std::endl;
        std::vector<unsigned> root(smart_index->getBaseLen(), 0);
        std::queue<unsigned> nextLevel;

        std::cout << "-- ep: " << ep << std::endl;
        std::cout << "relaGroup: ";
        for (int g : relaGroup)
        {
            std::cout << g << ", ";
        }
        std::cout << std::endl;

        // 1. 对ep_做DFS，所有看到的点都标记root为ep_
        unsigned c1 = 1;
        root[ep] = 1;
        for (auto g : relaGroup)
        {
            const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
            for (unsigned i = 0; i < tmp_graph[ep].size(); i++)
            {
                unsigned id = tmp_graph[ep][i].id;
                if (root[id])
                    continue;
                nextLevel.push(id);
                root[id] = 1;
                c1++;
            }
        }
        std::cout << "nextLevel.size(): " << nextLevel.size() << std::endl;

        while (nextLevel.size())
        {
            unsigned explore = nextLevel.front();
            nextLevel.pop();
            unsigned id;
            for (auto g : relaGroup)
            {
                const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                for (unsigned i = 0; i < tmp_graph[explore].size(); i++)
                {
                    id = tmp_graph[explore][i].id;
                    if (root[id])
                        continue;
                    nextLevel.push(id);
                    root[id] = 1;
                    c1++;
                }
            }
        }

        // 2. 看一下有多少个点没有办法通过ep_达到
        unsigned count = 0;
        for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
        {
            if (root[i] == 0)
            {
                isolate_ids.insert(i);
            }
        }
        std::cout << "##################################################" << std::endl;
        std::cout << "There are " << isolate_ids.size() << " of " << smart_index->getBaseLen() << " objects CANNOT be arrived from ep: " << ep << std::endl;
        std::cout << "Connected: " << c1 << " = (smart_index->getBaseLen() - isolate_ids.size()) = " << (smart_index->getBaseLen() - isolate_ids.size()) << std::endl;
        std::cout << "##################################################" << std::endl;

        if (isolate_ids.size())
        {
            return true;
        }
        return false;
    }

    void ComponentRelaConnectEnforcer::ConnectAndSpread(int group, const std::vector<int> relaGroup, std::set<unsigned> &isolate_ids)
    {
        bool printConnecting = false;
        {
            std::vector<int> groupList = {group};
            smart_index->summaryFinalGraph(groupList, {0, 1, smart_index->getBaseLen() - 1});
        }

        // --- spread connectivity ---
        std::vector<unsigned> root(smart_index->getBaseLen(), 0);
        std::queue<unsigned> nextLevel;
        unsigned numNoPos = 0, numUnconnected = isolate_ids.size();
        {
            auto &tmp_graph = smart_index->getFinalGraph(group);
            unsigned R = smart_index->getRRefineList()[group];
            for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
            {
                if (tmp_graph[i].size() == R)
                {
                    numNoPos++;
                }
            }
        }
        std::cout << "\n --- In index-group-" << group << ": numUnconnected(isolate points): " << numUnconnected << ", numNoPos(ids without addition out-neighbor space): " << numNoPos << std::endl;

        {
            while (isolate_ids.size())
            {
                // --- find node to add edge (connect) ---
                unsigned to = *isolate_ids.begin();
                SmartIndex::SimpleNeighbor from = get_connect_neighbor_limited_randEP(to, group, relaGroup, numNoPos, numUnconnected);
                unsigned from_id = from.id;
                from.id = to;
                if (from_id >= smart_index->getBaseLen())
                {
                    std::cerr << "❌ FATAL: from_id " << from_id << " out of bound (baseLen=" << smart_index->getBaseLen() << ")" << std::endl;
                    exit(-1);
                }
                smart_index->getFinalGraph(group)[from_id].emplace_back(from);
                if (printConnecting) {
                    std::cout << "connect from: " << from_id << " to: " << to << std::endl;
                }

                isolate_ids.erase(to);

                // 对to做DFS
                root[to] = 1;
                for (auto g : relaGroup)
                {
                    const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                    for (unsigned i = 0; i < tmp_graph[to].size(); i++)
                    {
                        unsigned id = tmp_graph[to][i].id;
                        if (root[id])
                            continue;
                        nextLevel.push(id);
                        root[id] = 1;
                        isolate_ids.erase(id);
                    }
                }
                while (nextLevel.size())
                {
                    unsigned explore = nextLevel.front();
                    nextLevel.pop();
                    unsigned id;
                    for (auto g : relaGroup)
                    {
                        const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                        for (unsigned i = 0; i < tmp_graph[explore].size(); i++)
                        {
                            id = tmp_graph[explore][i].id;
                            if (root[id])
                                continue;
                            nextLevel.push(id);
                            root[id] = 1;
                            isolate_ids.erase(id);
                        }
                    }
                }

                numUnconnected = isolate_ids.size();
            }
        }
        {
            std::vector<int> groupList = {group};
            smart_index->summaryFinalGraph(groupList, {0, 1, smart_index->getBaseLen() - 1});
        }
    }

    SmartIndex::SimpleNeighbor ComponentRelaConnectEnforcer::get_connect_neighbor_limited_randEP(unsigned query, int group, std::vector<int> relaGroup, unsigned &numNoPos, unsigned &numUnconnected, TYPE dist_type)
    {
        // --- # 3. 均没有（i.e.）不存在空位了，则扩大L，R，再进行（1，2）。 -----------------------------------------
        if ((numNoPos + std::min((unsigned)(smart_index->getBaseLen() * 0.01), (unsigned)1000)) >= smart_index->getBaseLen())
        {
            numNoPos = 0;
            unsigned delta = (numUnconnected - 1) / smart_index->getBaseLen() + 1;
            smart_index->getLRefineList()[group] += delta;
            smart_index->getRRefineList()[group] += delta;
            std::cout << "【没有添加位置了，将这个field的L和R都调大" << delta << "】, now, L = " << smart_index->getLRefineList()[group] << ", R = " << smart_index->getRRefineList()[group] << std::endl;
        }

        unsigned L = smart_index->getLRefineList()[group];
        unsigned R = smart_index->getRRefineList()[group];

        std::priority_queue<SmartIndex::Neighbor, std::vector<SmartIndex::Neighbor>, std::greater<SmartIndex::Neighbor>> cand_unchecked;
        std::priority_queue<SmartIndex::Neighbor, std::vector<SmartIndex::Neighbor>, std::less<SmartIndex::Neighbor>> result;
        std::priority_queue<SmartIndex::Neighbor, std::vector<SmartIndex::Neighbor>, std::less<SmartIndex::Neighbor>> result_withPos;

        boost::dynamic_bitset<> cal_flags{smart_index->getBaseLen(), 0};
        boost::dynamic_bitset<> check_flags{smart_index->getBaseLen(), 1}; 

        // --- # 1. 找在GreedySearch过程中所有被计算过距离的点中还有位置中距离最小的；-------------------------------------------------------------------------
        for (unsigned g : relaGroup)
        {
            unsigned ep = smart_index->each_ep_[g];
            while (ep == query)
                ep = rand() % smart_index->getBaseLen();
            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                      smart_index->getBaseDataList(), ep,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);
            result.emplace(SmartIndex::Neighbor(ep, dist, true));
            if (smart_index->getFinalGraph(group)[ep].size() < R)
            {
                result_withPos.emplace(SmartIndex::Neighbor(ep, dist, true));
            }
            cal_flags[ep] = true;
        }

        std::vector<unsigned> init_ids;

        cal_flags[query] = true;
        {
            unsigned ep = smart_index->each_ep_[group];
            check_flags[ep] = false;
            for (auto g : relaGroup)
            {
                const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                for (unsigned i = 0; i < tmp_graph[ep].size(); i++)
                {
                    unsigned id = tmp_graph[ep][i].id;
                    if (cal_flags[id])
                        continue;
                    init_ids.emplace_back(id);
                    cal_flags[id] = true;
                }
            }
        }
        while (init_ids.size() < L)
        {
            unsigned id = rand() % smart_index->getBaseLen();
            if (id == query)
                continue;
            if (cal_flags[id])
                continue;
            init_ids.emplace_back(id);
            cal_flags[id] = true;
        }

        for (unsigned i = 0; i < init_ids.size(); i++)
        {
            unsigned id = init_ids[i];
            if (id >= smart_index->getBaseLen())
                continue;
            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                      smart_index->getBaseDataList(), id,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);
            cand_unchecked.emplace(SmartIndex::Neighbor(id, dist, true));
            result.emplace(SmartIndex::Neighbor(id, dist, true));
            if (smart_index->getFinalGraph(group)[id].size() < R)
            {
                result_withPos.emplace(SmartIndex::Neighbor(id, dist, true));
            }
        }
        while (result.size() > L)
        {
            result.pop();
        }

        while (result.top().distance > cand_unchecked.top().distance && cand_unchecked.size() != 0)
        {
            unsigned n = cand_unchecked.top().id;
            cand_unchecked.pop();

            if (check_flags[n])
            {
                check_flags[n] = false;

                for (auto g : relaGroup)
                {
                    const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                    for (unsigned m = 0; m < tmp_graph[n].size(); m++)
                    {
                        unsigned id = tmp_graph[n][m].id;
                        if (id == query)
                            continue;
                        if (cal_flags[id])
                            continue;

                        cal_flags[id] = 1;
                        float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                                  smart_index->getBaseDataList(), id,
                                                                                  smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                                  smart_index->getGroupRepreList()[group], dist_type);
                        SmartIndex::Neighbor nn(id, dist, true);
                        cand_unchecked.emplace(nn);
                        if (result.size() < L)
                        {
                            result.emplace(nn);
                        }
                        else
                        {
                            if (dist >= result.top().distance)
                                continue;
                            else
                            {
                                result.emplace(nn);
                                result.pop();
                            }
                        }
                        if (smart_index->getFinalGraph(group)[id].size() < R)
                        {
                            result_withPos.emplace(SmartIndex::Neighbor(id, dist, true));
                        }
                    }
                }
            }
        }

        if (result_withPos.size())
        {
            while (result_withPos.size() != 1)
            {
                auto top = result_withPos.top();
                result_withPos.pop();
            }

            if (smart_index->getFinalGraph(group)[result_withPos.top().id].size() + 1 == R)
            {
                numNoPos++;
            }
            numUnconnected--;
            return SmartIndex::SimpleNeighbor(result_withPos.top().id, result_withPos.top().distance);
        }

        // --- # 2. 随机出一个有位置的ep来进行搜索. ---------------------------------------------------------------------------------------------
        {
            result = {};
            cand_unchecked = {};
            cal_flags = boost::dynamic_bitset<>(smart_index->getBaseLen(), 0);
            check_flags = boost::dynamic_bitset<>(smart_index->getBaseLen(), 1); // 是否还可以根据这个点进行拓展

            unsigned ep = rand() % smart_index->getBaseLen();
            while (smart_index->getFinalGraph(group)[ep].size() < R)
            {
                ep = rand() % smart_index->getBaseLen();
            }
            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                      smart_index->getBaseDataList(), ep,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);
            result.emplace(SmartIndex::Neighbor(ep, dist, true));
            result_withPos.emplace(SmartIndex::Neighbor(ep, dist, true));
            cal_flags[ep] = true;

            std::vector<unsigned> init_ids;

            cal_flags[query] = true;
            {
                unsigned ep = smart_index->each_ep_[group];
                check_flags[ep] = false;
                for (auto g : relaGroup)
                {
                    const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                    for (unsigned i = 0; i < tmp_graph[ep].size(); i++)
                    {
                        unsigned id = tmp_graph[ep][i].id;
                        if (cal_flags[id])
                            continue;
                        init_ids.emplace_back(id);
                        cal_flags[id] = true;
                    }
                }
            }
            while (init_ids.size() < L)
            {
                unsigned id = rand() % smart_index->getBaseLen();
                if (id == query)
                    continue;
                if (cal_flags[id])
                    continue;
                init_ids.emplace_back(id);
                cal_flags[id] = true;
            }

            for (unsigned i = 0; i < init_ids.size(); i++)
            {
                unsigned id = init_ids[i];
                if (id >= smart_index->getBaseLen())
                    continue;
                float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                          smart_index->getBaseDataList(), id,
                                                                          smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                          smart_index->getGroupRepreList()[group], dist_type);
                cand_unchecked.emplace(SmartIndex::Neighbor(id, dist, true));
                result.emplace(SmartIndex::Neighbor(id, dist, true));
                if (smart_index->getFinalGraph(group)[id].size() < R)
                {
                    result_withPos.emplace(SmartIndex::Neighbor(id, dist, true));
                }
            }
            while (result.size() > L)
            {
                result.pop();
            }

            while (result.top().distance > cand_unchecked.top().distance && cand_unchecked.size() != 0)
            {
                unsigned n = cand_unchecked.top().id;
                cand_unchecked.pop();

                if (check_flags[n])
                {
                    check_flags[n] = false;

                    for (auto g : relaGroup)
                    {
                        const weavess::SmartIndex::FinalGraph &tmp_graph = smart_index->getFinalGraph(g);
                        for (unsigned m = 0; m < tmp_graph[n].size(); m++)
                        {
                            unsigned id = tmp_graph[n][m].id;
                            if (id == query)
                                continue;
                            if (cal_flags[id])
                                continue;

                            cal_flags[id] = 1;
                            float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), query,
                                                                                      smart_index->getBaseDataList(), id,
                                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                                      smart_index->getGroupRepreList()[group], dist_type);
                            SmartIndex::Neighbor nn(id, dist, true);
                            cand_unchecked.emplace(nn);
                            if (result.size() < L)
                            {
                                result.emplace(nn);
                            }
                            else
                            {
                                if (dist >= result.top().distance)
                                    continue;
                                else
                                {
                                    result.emplace(nn);
                                    result.pop();
                                }
                            }
                            if (smart_index->getFinalGraph(group)[id].size() < R)
                            {
                                result_withPos.emplace(SmartIndex::Neighbor(id, dist, true));
                            }
                        }
                    }
                }
            }

            while (result_withPos.size() != 1)
            {
                auto top = result_withPos.top();
                result_withPos.pop();
            }

            if (smart_index->getFinalGraph(group)[result_withPos.top().id].size() + 1 == R)
            {
                numNoPos++;
            }
            numUnconnected--;
            return SmartIndex::SimpleNeighbor(result_withPos.top().id, result_withPos.top().distance);
        }
    }
}