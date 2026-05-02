//
// Created by mengtong-x on 2026/04/30.
//

#include "component.h"


// ======================================
// Initialization
//      For: init()
// ======================================
namespace weavess {

    // ----------------------------------------------------------------------------------------------------
    // initialization -- preliminary
    // ----------------------------------------------------------------------------------------------------

    /** 2025.06.20
     * ComponentPreliminary_smart::SeperateInner_smart_GroupEquNoTotal()
     *      For "vamana_equNoTotal": related group = itself
     */
    void ComponentPreliminary_smart::SeperateInner_smart_GroupEquNoTotal(Parameters &parameters)
    {
        // ## --------- 对应每组传入representative vector进行indexing -----------------------------------------
        std::vector<std::vector<int>> groupList;      // 用于存储哪些 field 不为0 (即 proto 中非0 field 索引)
        std::vector<std::vector<int>> relaList;       // 每组需要 reference 的其他组
        std::vector<std::vector<float>> group_protos; // 原始 prototype，float 存储
        // -- single --
        for (int k = 0; k < smart_index->getFieldNum(); ++k)
        {
            // === 准备 PROTO
            std::vector<float> proto(smart_index->getFieldNum(), 0.0);
            proto[k] = 1.0;
            group_protos.emplace_back(proto);

            // === 准备 RELA
            std::vector<int> rela(1);
            rela[0] = k;
            relaList.emplace_back(rela);

            // === 准备 groupList: proto 中 field > 0 的索引
            std::vector<int> active_fields;
            active_fields.emplace_back(k);
            groupList.emplace_back(active_fields);

            // === 打印信息
            std::cout << "Group " << k << ": ";
            std::cout << "PROTO: ";
            for (auto p : proto)
                std::cout << p << " ";
            std::cout << " | RELA: ";
            for (auto r : rela)
                std::cout << r << " ";
            std::cout << " | Active Fields: ";
            for (auto f : active_fields)
                std::cout << f << " ";
            std::cout << std::endl;
        }

        // ## set up: related to group
        smart_index->setGroupList(groupList);
        smart_index->setGroupRepreList(group_protos);
        smart_index->setRelaCheckList(relaList);
        // ------------------------------------------------------------------------------------------

        // ## resize Graph Lists space
        smart_index->getFinalGraphList().resize(smart_index->getGroupNum());
        smart_index->getLoadGraphList().resize(smart_index->getGroupNum());
        smart_index->getExactGraphList().resize(smart_index->getGroupNum());
        // ## resize other parameter space (related to Group Num)
        smart_index->each_ep_.resize(smart_index->getGroupNum());
    }

    /** 2025.06.20
     * ComponentPreliminary_smart::SeperateInner_smart_GroupFusion()
     *      For "vamana_fusion": related group = itself
     */
    void ComponentPreliminary_smart::SeperateInner_smart_GroupFusion(Parameters &parameters)
    {
        std::vector<std::vector<int>> groupList, relaList;
        std::vector<int> tmp, tmpRela = {0};

        // ## total field
        for (int f = 0; f < smart_index->getFieldNum(); f++)
        {
            tmp.emplace_back(f);
        }
        groupList.emplace_back(tmp);
        relaList.emplace_back(tmpRela);
        std::vector<std::vector<float>> group_protos; // 原始 prototype，float 存储
        std::vector<float> proto(smart_index->getFieldNum(), 1.0);
        group_protos.emplace_back(proto);

        // ## set up: related to group
        smart_index->setGroupList(groupList);
        smart_index->setRelaCheckList(relaList);
        smart_index->setGroupRepreList(group_protos);

        // ## resize Graph Lists space
        smart_index->getFinalGraphList().resize(smart_index->getGroupNum());
        smart_index->getLoadGraphList().resize(smart_index->getGroupNum());
        smart_index->getExactGraphList().resize(smart_index->getGroupNum());
        // ## resize other parameter space (related to Group Num)
        smart_index->each_ep_.resize(smart_index->getGroupNum());
    }

    /** 2026.01.24
     * ComponentPreliminary_smart::SeperateInner_smart_GroupAllWeight()
     *      For "vamana_equNoTotal": related group = itself
     */
    void ComponentPreliminary_smart::SeperateInner_smart_GroupAllWeight(Parameters &parameters)
    {
        // ## --------- 对应每组传入representative vector进行indexing -----------------------------------------
        std::vector<std::vector<int>> groupList;      // 用于存储哪些 field 不为0 (即 proto 中非0 field 索引)
        std::vector<std::vector<int>> relaList;       // 每组需要 reference 的其他组
        std::vector<std::vector<float>> group_protos; // 原始 prototype，float 存储


        std::string txt_path = "../dataset/useWeight/useWeightEachQuery.txt";
        std::ifstream file(txt_path);
        std::string line;

        std::vector<float> weight(smart_index->getFieldNum());
        unsigned numComb;
        bool generate_all_weight = false;

        if (file.is_open())
        {
            file >> numComb;
            if (numComb == 0)
            {
                std::cout << "___【 Warning! 】: " << txt_path << "No weight provided! Degrade to use weavess::TYPE::ALL_WEIGHT ___" << std::endl;
                generate_all_weight = true;
            }
            for (unsigned i = 0; i < numComb; i++)
            {
                for (int j = 0; j < smart_index->getFieldNum(); j++)
                {
                    file >> weight[j];
                }
                group_protos.emplace_back(weight);
            }
        }
        else
        {
            std::cout << "___【 Warning! 】: " << txt_path << "does't exist! Degrade to use weavess::TYPE::ALL_WEIGHT ___" << std::endl;
            generate_all_weight = true;
        }
        
        if (generate_all_weight) {
            for (int w = 1; w < (1 << smart_index->getFieldNum()); ++w)
            {   // 2^size = (1 << size)
                for (size_t j = 0; j < smart_index->getFieldNum(); ++j)
                {
                    weight[j] = (w & (1 << j)) ? 1.0f : 0.0f; // 第 j 位
                }
                group_protos.emplace_back(weight);
                if (true)
                {
                    // 打印当前组合
                    std::cout << "Combination weight " << w << " / " << ((1 << smart_index->getFieldNum()) - 1) << " : ";
                    for (float ww : weight)
                    {
                        std::cout << ww << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        
        for (int k = 0; k < group_protos.size(); ++k)
        {
            // === 准备 PROTO
            std::vector<float> proto = group_protos[k];

            // === 准备 RELA
            std::vector<int> rela(1);
            rela[0] = k;
            relaList.emplace_back(rela);

            // === 准备 groupList: proto 中 field > 0 的索引
            std::vector<int> active_fields;
            for (int i = 0; i < smart_index->getFieldNum(); i++) {
                if (proto[i] != 0) { active_fields.emplace_back(i); }
            }
            groupList.emplace_back(active_fields);

            // === 打印信息
            std::cout << "Group " << k << ": ";
            std::cout << "PROTO: ";
            for (auto p : proto)
                std::cout << p << " ";
            std::cout << " | RELA: ";
            for (auto r : rela)
                std::cout << r << " ";
            std::cout << " | Active Fields: ";
            for (auto f : active_fields)
                std::cout << f << " ";
            std::cout << std::endl;
        }

        // ## set up: related to group
        smart_index->setGroupList(groupList);
        smart_index->setGroupRepreList(group_protos);
        smart_index->setRelaCheckList(relaList);
        // ------------------------------------------------------------------------------------------

        // ## resize Graph Lists space
        smart_index->getFinalGraphList().resize(smart_index->getGroupNum());
        smart_index->getLoadGraphList().resize(smart_index->getGroupNum());
        smart_index->getExactGraphList().resize(smart_index->getGroupNum());
        // ## resize other parameter space (related to Group Num)
        smart_index->each_ep_.resize(smart_index->getGroupNum());
    }

    /** 2025.05.06
     * ComponentPreliminary_smart::SeperateInner_smart_ClusterGroup()
     *      1. 划分field分组。在后续程序中，每一个分组会对应一个graph；
     *      2. 根据group数，resize各种GraphList
     */
    void ComponentPreliminary_smart::SeperateInner_smart_ClusterGroup(Parameters &parameters)
    {
        std::vector<std::vector<int>> groupList;      // 用于存储哪些 field 不为0 (即 proto 中非0 field 索引)
        std::vector<std::vector<int>> relaList;       // 每组需要 reference 的其他组
        std::vector<std::vector<int>> member_list;    // 原始成员（可选）
        std::vector<std::vector<float>> group_protos; // 原始 prototype，float 存储

        // ### --- 从文件中读取 group 信息 ---
        std::string file = "../include/python_file/cluster_groups.txt";
        std::ifstream infile(file);
        if (!infile)
        {
            std::cerr << "❌ Cannot open file: " << file << std::endl;
            return;
        }

        int K;
        infile >> K;
        std::string line;
        std::getline(infile, line); // skip rest of first line

        for (int k = 0; k < K; ++k)
        {
            std::getline(infile, line); // # Group k

            // === 读取 PROTO
            std::getline(infile, line); // PROTO:
            std::istringstream proto_ss(line.substr(7));
            std::vector<float> proto;
            float val;
            while (proto_ss >> val)
                proto.push_back(val);
            group_protos.push_back(proto);

            // === 读取 MEMBERS
            std::getline(infile, line); // MEMBERS:
            std::istringstream members_ss(line.substr(9));
            std::vector<int> members;
            while (members_ss >> val)
                members.push_back(val);
            member_list.push_back(members);

            // === 读取 RELA
            std::getline(infile, line); // RELA:
            std::istringstream rela_ss(line.substr(6));
            std::vector<int> rela;
            while (rela_ss >> val)
                rela.push_back(val);
            relaList.push_back(rela);

            // === 生成 groupList: proto 中 field > 0 的索引
            std::vector<int> active_fields;
            for (size_t f = 0; f < proto.size(); ++f)
            {
                if (proto[f] > 0.0f)
                    active_fields.push_back(f);
            }
            groupList.push_back(active_fields);

            // === 打印信息
            std::cout << "Group " << k << ": ";
            std::cout << "PROTO: ";
            for (auto p : proto)
                std::cout << p << " ";
            std::cout << " | MEMBERS: ";
            for (auto m : members)
                std::cout << m << " ";
            std::cout << " | RELA: ";
            for (auto r : rela)
                std::cout << r << " ";
            std::cout << " | Active Fields: ";
            for (auto f : active_fields)
                std::cout << f << " ";
            std::cout << std::endl;
        }
        std::cout << "✅ Loaded " << K << " groups from " << file << std::endl;

        // ## set up: related to group
        smart_index->setGroupList(groupList);
        smart_index->setGroupRepreList(group_protos);
        smart_index->setRelaCheckList(relaList);

        // ## resize Graph Lists space
        smart_index->getFinalGraphList().resize(smart_index->getGroupNum());
        smart_index->getLoadGraphList().resize(smart_index->getGroupNum());
        smart_index->getExactGraphList().resize(smart_index->getGroupNum());
        // ## resize other parameter space (related to Group Num)
        smart_index->each_ep_.resize(smart_index->getGroupNum());
    }

    /** 
     * ComponentPreliminary_smart::PrepareParameter_smart_Equal()
     *      给每个group分配L_refine，R_refine。
     *      分配方式：平均分L_refine，R_refine； = L_refine/numGroup
     *      传入alpha2
     */
    void ComponentPreliminary_smart::PrepareParameter_smart_Equal(Parameters &parameters)
    {
        unsigned L = parameters.get<unsigned>("L_refine"), R = parameters.get<unsigned>("R_refine");
        std::vector<unsigned> param(smart_index->getGroupNum());

        // -- 均分&向上取整
        unsigned aveSpace =  (unsigned) std::ceil( (float)L / (float)smart_index->getGroupList().size() );
        for (int i = 0; i < smart_index->getGroupNum(); i++)
        {
            param[i] = aveSpace;
        }
        smart_index->setLRefineList(param);

        aveSpace = (unsigned) std::ceil( (float)R / (float)smart_index->getGroupList().size() );
        for (int i = 0; i < smart_index->getGroupNum(); i++)
        {
            param[i] = aveSpace;
        }
        smart_index->setRRefineList(param);

        // alpha2
        smart_index->setAlpha2(parameters.get<float>("alpha2"));
    }









    // ----------------------------------------------------------------------------------------------------
    // initialization -- initial graph
    // ----------------------------------------------------------------------------------------------------
    // -------------------------------------------------------
    // For PolyGraph and Vamana-series: init()
    // -------------------------------------------------------
    void ComponentInitRand_smart::SetConfigs_smart(int group)
    {
        smart_index->R = smart_index->getRRefineList()[group];
        smart_index->L = smart_index->getLRefineList()[group];
        std::cout << "smart_index->L: " << smart_index->L << std::endl;
        std::cout << "smart_index->R: " << smart_index->R << std::endl;
    }

    void ComponentInitRand_smart::InitInner_smart(TYPE dist_type)
    {
        unsigned R_refine = 0, L_refine = 0;
        for (int i = 0; i < smart_index->getGroupNum(); i++) {
            R_refine += smart_index->getRRefineList()[i];
            L_refine += smart_index->getLRefineList()[i];
        }
        smart_index->getParam().set<unsigned>("R_refine", R_refine);
        smart_index->getParam().set<unsigned>("L_refine", L_refine);
        std::cout << "smart_index->L_refine: " << smart_index->getParam().get<unsigned>("L_refine") << std::endl;
        std::cout << "smart_index->R_refine: " << smart_index->getParam().get<unsigned>("R_refine") << std::endl;

        // std::mt19937 rng(rand());
        std::mt19937 rng(555);
        std::vector<std::vector<unsigned>> initON(smart_index->getBaseLen(), std::vector<unsigned>(smart_index->getParam().get<unsigned>("R_refine")));

        for (unsigned i = 0; i < initON.size(); i++)
        {
            weavess::GenRandom(rng, initON[i].data(), smart_index->getParam().get<unsigned>("R_refine"), smart_index->getBaseLen());
        }
        // #pragma omp parallel for
        for (int group = 0; group < smart_index->getGroupNum(); group++)
        {
            std::cout << "__START INIT(4SMART) : RAND for group " << group << "__" << std::endl;
            std::cout << "  with group element: ";
            for (auto f : smart_index->getGroupList()[group])
            {
                std::cout << f << ", ";
            }
            std::cout << std::endl;
            ComponentInitRand_smart::InitInner_smart_4group(group, rng, initON, dist_type);
            std::cout << "__END INIT(4SMART) : RAND for group " << group << "__" << std::endl;
        }

        std::vector<std::vector<unsigned>>().swap(initON);
    }

    void ComponentInitRand_smart::InitInner_smart_4group(int group, std::mt19937 &rng, std::vector<std::vector<unsigned>> &initON, TYPE dist_type)
    {
        SetConfigs_smart(group);
        std::vector<int> groupElement = smart_index->getGroupList()[group];

        smart_index->graph_.resize(smart_index->getBaseLen());
        unsigned starPos = 0;
        for (int i = 0; i < group; i++)
        {
            starPos += smart_index->getRRefineList()[i];
        }

#pragma omp parallel for
        for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
        {
            std::vector<unsigned> tmp(smart_index->R);
            for (unsigned j = 0, pos = starPos; j < tmp.size(); j++, pos++)
            {
                tmp[j] = initON[i][pos];
            }

            for (unsigned j = 0; j < smart_index->R; j++)
            {
                unsigned id = tmp[j];
                if (group == 4 && i == 0 && j == 0)
                {
                    std::cout << "Cal for dist-total" << std::endl;
                }

                while (id == i || id >= smart_index->getBaseLen())
                {
                    id = rand() % smart_index->getBaseLen();
                }

                float dist = smart_index->getDist()->compare_smart_weight(smart_index->getBaseDataList(), i,
                                                                          smart_index->getBaseDataList(), id,
                                                                          smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                          smart_index->getGroupRepreList()[group], dist_type);
                smart_index->graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(smart_index->graph_[i].pool.begin(), smart_index->graph_[i].pool.end());
            smart_index->graph_[i].pool.reserve(smart_index->R);
        }

        smart_index->getFinalGraph(group).resize(smart_index->getBaseLen());
        for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
        {
            std::vector<SmartIndex::SimpleNeighbor> tmp;

            std::sort(smart_index->graph_[i].pool.begin(), smart_index->graph_[i].pool.end());

            for (auto &j : smart_index->graph_[i].pool)
            {
                tmp.push_back(SmartIndex::SimpleNeighbor(j.id, j.distance));
            }

            smart_index->getFinalGraph(group)[i] = tmp;

            std::vector<SmartIndex::Neighbor>().swap(smart_index->graph_[i].pool);
            std::vector<unsigned>().swap(smart_index->graph_[i].nn_new);
            std::vector<unsigned>().swap(smart_index->graph_[i].nn_old);
            std::vector<unsigned>().swap(smart_index->graph_[i].rnn_new);
            std::vector<unsigned>().swap(smart_index->graph_[i].rnn_old);
        }

        std::vector<SmartIndex::nhood>().swap(smart_index->graph_);
    }


    // -------------------------------------------------------
    // For HNSW_Fusion: init() as construction
    // -------------------------------------------------------
    // -- build
    void ComponentInitHNSW_Fusion::InitInner_smart(TYPE dist_type)
    {
        SetConfigs();

        Build(false, dist_type);
    }

    void ComponentInitHNSW_Fusion::SetConfigs()
    {
        smart_index->max_m_ = smart_index->getParam().get<unsigned>("max_m");
        smart_index->m_ = smart_index->max_m_;
        smart_index->max_m0_ = smart_index->getParam().get<unsigned>("max_m0");
        auto ef_construction_ = smart_index->getParam().get<unsigned>("ef_construction");
        if (ef_construction_ > 0)
            smart_index->ef_construction_ = ef_construction_;
        smart_index->n_threads_ = smart_index->getParam().get<unsigned>("n_threads");
        smart_index->mult = smart_index->getParam().get<int>("mult");
        smart_index->level_mult_ = smart_index->mult > 0 ? smart_index->mult : (1 / log(1.0 * smart_index->m_));
    }

    void ComponentInitHNSW_Fusion::Build(bool reverse, TYPE dist_type)
    {
        smart_index->nodes_.resize(smart_index->getBaseLen());
        int level = GetRandomNodeLevel();
        auto *first = new SmartIndex::HnswNode(0, level, smart_index->max_m_, smart_index->max_m0_);
        smart_index->nodes_[0] = first;
        smart_index->max_level_ = level;
        smart_index->enterpoint_ = first;
        unsigned count = 0;
#pragma omp parallel
        {
            auto *visited_list = new SmartIndex::VisitedList(smart_index->getBaseLen());
#pragma omp for schedule(dynamic, 128)
            for (size_t i = 1; i < smart_index->getBaseLen(); ++i)
            {
                int level = GetRandomNodeLevel();
                auto *qnode = new SmartIndex::HnswNode(i, level, smart_index->max_m_, smart_index->max_m0_);
                smart_index->nodes_[i] = qnode;
                InsertNode(qnode, visited_list, dist_type);
                count++;
                if ((count) % 1000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process: " << count << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }

            delete visited_list;
        }
    }

    int ComponentInitHNSW_Fusion::GetRandomSeedPerThread()
    {
        int tid = omp_get_thread_num();
        int g_seed = 17;
        for (int i = 0; i <= tid; ++i)
            g_seed = 214013 * g_seed + 2531011;
        return (g_seed >> 16) & 0x7FFF;
    }

    int ComponentInitHNSW_Fusion::GetRandomNodeLevel()
    {
        static thread_local std::mt19937 rng(GetRandomSeedPerThread());
        static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
        double r = uniform_distribution(rng);

        if (r < std::numeric_limits<double>::epsilon())
            r = 1.0;
        return (int)(-log(r) * smart_index->level_mult_);
    }

    void ComponentInitHNSW_Fusion::InsertNode(SmartIndex::HnswNode *qnode, SmartIndex::VisitedList *visited_list, TYPE dist_type)
    {
        std::vector<int> need_calcu;
        for (int i = 0; i < smart_index->getFieldNum(); i++)
        {
            need_calcu.emplace_back(i);
        }

        int cur_level = qnode->GetLevel();
        std::unique_lock<std::mutex> max_level_lock(smart_index->max_level_guard_, std::defer_lock);
        if (cur_level > smart_index->max_level_)
            max_level_lock.lock();

        int max_level_copy = smart_index->max_level_;
        SmartIndex::HnswNode *enterpoint = smart_index->enterpoint_;

        if (cur_level < max_level_copy)
        {
            SmartIndex::HnswNode *cur_node = enterpoint;
            float d = smart_index->getDist()->compare_smart(smart_index->getBaseDataList(), qnode->GetId(),
                                                            smart_index->getBaseDataList(), cur_node->GetId(),
                                                            smart_index->getBaseDimList(), need_calcu,
                                                            TYPE::DIST_EUCLIDEAN);
            float cur_dist = d;
            for (auto i = max_level_copy; i > cur_level; --i)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    std::unique_lock<std::mutex> local_lock(cur_node->GetAccessGuard());
                    const std::vector<SmartIndex::HnswNode *> &neighbors = cur_node->GetFriends(i);

                    for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter)
                    {
                        d = smart_index->getDist()->compare_smart(smart_index->getBaseDataList(), qnode->GetId(),
                                                                  smart_index->getBaseDataList(), (*iter)->GetId(),
                                                                  smart_index->getBaseDimList(), need_calcu,
                                                                  TYPE::DIST_EUCLIDEAN);

                        if (d < cur_dist) 
                        {
                            cur_dist = d;
                            cur_node = *iter;
                            changed = true;
                        }
                    }
                }
            }
            enterpoint = cur_node;
        }

        // PRUNE
        ComponentPrune_smart *a = new ComponentPruneHeuristic_smart(smart_index);

        for (auto i = std::min(max_level_copy, cur_level); i >= 0; --i)
        {
            std::priority_queue<SmartIndex::FurtherFirst> result;
            SearchAtLayer(qnode, enterpoint, i, visited_list, result, dist_type);

            a->Hnsw2Neighbor_Fusion(qnode->GetId(), smart_index->m_, result, dist_type);

            while (!result.empty())
            {
                auto *top_node = result.top().GetNode();
                result.pop();
                Link(top_node, qnode, i, dist_type);
                Link(qnode, top_node, i, dist_type);
            }
        }
        if (cur_level > smart_index->enterpoint_->GetLevel())
        {
            smart_index->enterpoint_ = qnode;
            smart_index->max_level_ = cur_level;
        }
    }

    void ComponentInitHNSW_Fusion::SearchAtLayer(SmartIndex::HnswNode *qnode, SmartIndex::HnswNode *enterpoint, int level,
                                                 SmartIndex::VisitedList *visited_list,
                                                 std::priority_queue<SmartIndex::FurtherFirst> &result, TYPE dist_type)
    {
        std::vector<int> need_calcu;
        for (int i = 0; i < smart_index->getFieldNum(); i++)
        {
            need_calcu.emplace_back(i);
        }

        std::priority_queue<SmartIndex::CloserFirst> candidates;
        float d = smart_index->getDist()->compare_smart(smart_index->getBaseDataList(), qnode->GetId(),
                                                        smart_index->getBaseDataList(), enterpoint->GetId(),
                                                        smart_index->getBaseDimList(), need_calcu,
                                                        TYPE::DIST_EUCLIDEAN);
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

            for (const auto &neighbor : neighbors)
            {
                int id = neighbor->GetId();
                if (visited_list->NotVisited(id))
                {
                    visited_list->MarkAsVisited(id);
                    d = smart_index->getDist()->compare_smart(smart_index->getBaseDataList(), qnode->GetId(),
                                                              smart_index->getBaseDataList(), neighbor->GetId(),
                                                              smart_index->getBaseDimList(), need_calcu,
                                                              TYPE::DIST_EUCLIDEAN);
                    if (result.size() < smart_index->ef_construction_ || result.top().GetDistance() > d)
                    {
                        result.emplace(neighbor, d);
                        candidates.emplace(neighbor, d);
                        if (result.size() > smart_index->ef_construction_)
                            result.pop();
                    }
                }
            }
        }
    }

    void ComponentInitHNSW_Fusion::Link(SmartIndex::HnswNode *source, SmartIndex::HnswNode *target, int level, TYPE dist_type)
    {
        std::vector<int> need_calcu;
        for (int i = 0; i < smart_index->getFieldNum(); i++)
        {
            need_calcu.emplace_back(i);
        }

        std::unique_lock<std::mutex> lock(source->GetAccessGuard()); // 使用互斥锁确保在多线程环境下对源节点 source 的访问是线程安全的
        std::vector<SmartIndex::HnswNode *> &neighbors = source->GetFriends(level);
        neighbors.push_back(target);
        bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                      (level <= 0 && neighbors.size() > source->GetMaxM0());
        if (!shrink)
            return;

        std::priority_queue<SmartIndex::FurtherFirst> tempres;
        for (const auto &neighbor : neighbors)
        {
            float tmp = smart_index->getDist()->compare_smart(smart_index->getBaseDataList(), source->GetId(),
                                                              smart_index->getBaseDataList(), neighbor->GetId(),
                                                              smart_index->getBaseDimList(), need_calcu,
                                                              TYPE::DIST_EUCLIDEAN);
            tempres.push(SmartIndex::FurtherFirst(neighbor, tmp));
        }

        // PRUNE
        ComponentPrune_smart *a = new ComponentPruneHeuristic_smart(smart_index);
        a->Hnsw2Neighbor_Fusion(source->GetId(), tempres.size() - 1, tempres, dist_type);

        neighbors.clear();
        while (!tempres.empty())
        {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
        std::priority_queue<SmartIndex::FurtherFirst>().swap(tempres);
    }











    // ----------------------------------------------------------------------------------------------------
    // initialization -- entry-point selection
    // ----------------------------------------------------------------------------------------------------
    void ComponentRefineEntryCentroid_smart::EntryInner_smart()
    {
        std::cout << "__START ENTRY_INNER: Centroid__" << std::endl;
        center_.resize(smart_index->getFieldNum());
        auto s1 = std::chrono::high_resolution_clock::now();

        for (int f = 0; f < smart_index->getFieldNum(); f++)
        {
            center_[f] = new float[smart_index->getBaseDim(f)];
            for (unsigned j = 0; j < smart_index->getBaseDim(f); j++)
                center_[f][j] = 0.0;
            for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
            {
                for (unsigned j = 0; j < smart_index->getBaseDim(f); j++)
                {
                    center_[f][j] += smart_index->getBaseData(f)[i * smart_index->getBaseDim(f) + j];
                }
            }

            for (unsigned j = 0; j < smart_index->getBaseDim(f); j++)
            {
                center_[f][j] /= (smart_index->getBaseLen() + 0.0);
            }
        }

        std::vector<unsigned> ep_list_(smart_index->getFieldNum());
        for (unsigned g = 0; g < smart_index->getGroupNum(); g++)
        {
            EntryInner_smart_4group(g);
        }
        auto e1 = std::chrono::high_resolution_clock::now();

        std::cout << "\n========== Enter-points For Each Group ========================" << std::endl;
        std::cout << "each_ep_: ";
        for (unsigned g = 0; g < smart_index->getGroupNum(); g++)
        {
            std::cout << smart_index->each_ep_[g] << ", ";
        }
        std::cout << "\n==============================================" << std::endl;
        std::chrono::duration<double> load_info_time = e1 - s1;
        std::cout << "__END: ENTRY_INNER: Centroid__" << std::endl;
        std::cout << "^^^^^^ EntryInner() time is " << load_info_time.count() << " ^^^^^^ \n"
                  << std::endl;
    }

    void ComponentRefineEntryCentroid_smart::EntryInner_smart_4group(int group)
    {
        SmartIndex::Neighbor nn;
        get_exact_neighbor_smart_4group(group, nn);
        smart_index->each_ep_[group] = nn.id;

        std::cout << "--------------------------------------" << std::endl;
        std::cout << "In group " << group << ", Nearest Neighbor to centriod is: " << nn.id << ", with distance: " << nn.distance << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

    void ComponentRefineEntryCentroid_smart::get_exact_neighbor_smart_4group(int group, SmartIndex::Neighbor &nn, TYPE dist_type)
    {
        nn.distance = MAXFLOAT;

        smart_index->each_ep_[group] = rand() % smart_index->getBaseLen(); // random initialize navigating point
        for (unsigned i = 0; i < smart_index->getBaseLen(); i++)
        {
            float dist = smart_index->getDist()->compare_smart_weight(center_, 0,
                                                                      smart_index->getBaseDataList(), i,
                                                                      smart_index->getBaseDimList(), smart_index->getGroupList()[group],
                                                                      smart_index->getGroupRepreList()[group], dist_type);
            if (dist < nn.distance)
            {
                nn.distance = dist;
                nn.id = i;
            }
        }
        smart_index->each_ep_[group] = nn.id;
    }
}