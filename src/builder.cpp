//
// Created by mengtong-x on 2024/06/23.
//

#include "builder.h"
#include "component.h"
#include <set>
#include <fstream>
#include <string>


# include <naive/search.h>

namespace weavess
{
    // -----------------------------------------------------------------
    // For SmartIndex
    // -----------------------------------------------------------------
    /**
     * load dataset, parameters
     * @param data_file *_base.fvecs
     * @param query_file *_query.fvecs
     * @param ground_file *_groundtruth.ivecs
     * @param parameters
     * @return pointer of builder
     */
    SmartIndexBuilder *SmartIndexBuilder::load(Parameters &parameters)
    {
        auto s1 = std::chrono::high_resolution_clock::now();

        auto *a = new ComponentLoad_smart(smart_final_index_);

        // ## read data
        std::string alg = parameters.get<std::string>("alg");
        if (alg == "Smart" || alg == "hnsw_fusion" || alg == "vamana_equNoTotal" || alg == "vamana_fusion" || alg == "vamana_allWeight" || alg == "vamana_oracle"
        )
        {
            a->LoadInner_smart(parameters);
        }
        else
        {
            std::cout << "In SmartIndexBuilder::load(), meet alg that don't know how to handle: " << alg << std::endl;
            exit(-1);
        }

        auto e1 = std::chrono::high_resolution_clock::now();
        std::cout << "==============================================" << std::endl;
        std::cout << "__LOAD FINISH__" << std::endl;
        std::chrono::duration<double> load_info_time = e1 - s1;
        std::cout << "^^^^^^ Individual load() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        // ## Show data summary information
        std::cout << "========== Data Information Summary ========================" << std::endl;
        std::cout << "base data len : " << smart_final_index_->getBaseLen() << std::endl;
        std::cout << "query data len : " << smart_final_index_->getQueryLen() << std::endl;
        std::cout << "ground truth data len : " << smart_final_index_->getGroundLen() << std::endl;
        std::cout << "ground truth data dim (i.e. K) : " << smart_final_index_->getGroundDim() << std::endl;
        for (int f = 0; f < smart_final_index_->getFieldNum(); f++)
        {
            std::cout << "-----------------------" << std::endl;
            std::cout << "- In field " << (f + 1) << std::endl;
            std::cout << "base data dim : " << smart_final_index_->getBaseDim(f) << std::endl;
            std::cout << "query data dim : " << smart_final_index_->getQueryDim(f) << std::endl;
        }
        std::cout << "==============================================" << std::endl;
        // std::cout << smart_final_index_->getParam().toString() << std::endl;
        smart_final_index_->getParam().showAllParams();
        std::cout << "==============================================\n\n\n" << std::endl;

        return this;
    }



    /**
     * 1. load search-weights
     * @param data_file *_base.fvecs
     * @param query_file *_query.fvecs
     * @param ground_file *_groundtruth.ivecs
     * @param parameters
     * @return pointer of builder
     */
    SmartIndexBuilder *SmartIndexBuilder::load_search_weight()
    {
        auto s1 = std::chrono::high_resolution_clock::now();

        std::string txt_path = "../dataset/useWeight/useWeightEachQuery.txt";
        std::string dataset = smart_final_index_->getParam().get<std::string>("dataset");
        if (dataset == "New")
        {
            txt_path = "../dataset/useWeight/useWeightEachQuery_New.txt";
        }
        else if (dataset == "ImageText")
        {
            txt_path = "../dataset/useWeight/useWeightEachQuery_2fields_IT.txt";
        }
        else if (dataset == "QA" || dataset == "QA2")
        {
            txt_path = "../dataset/useWeight/useWeightEachQuery_4fields_QA.txt";
        }
        else if (dataset == "Wiki")
        {
            txt_path = "../dataset/useWeight/useWeightEachQuery_6fields_Wiki.txt";
        }
        else if (dataset == "Protein")
        {
            txt_path = "../dataset/useWeight/useWeightEachQuery_8fields_Protein.txt";
        }
        else
        {
            std::cout << "dataset error!\n";
            exit(-1);
        }
        std::cout << "___ Read useWeightEachQuery.txt from: " << txt_path << " ___" << std::endl;
        std::ifstream file(txt_path);

        std::vector<float> weight(smart_final_index_->getFieldNum());
        std::vector<std::vector<float>> allWeight;
        unsigned numComb;
        int field_num = smart_final_index_->getFieldNum();

        if (file.is_open())
        {
            file >> numComb;
            if (numComb == 0)
            {
                std::cout << "___【 Warning! 】: " << txt_path << "; No weight provided! Set weight empty to Degrade to use weavess::TYPE::ALL_WEIGHT ___" << std::endl;
                getFinalIndex()->setSearchWeight(allWeight);
            }
            else {
                for (unsigned i = 0; i < numComb; i++)
                {
                    for (int j = 0; j < field_num; j++)
                    {
                        file >> weight[j];
                    }
                    allWeight.emplace_back(weight);
                }
                getFinalIndex()->setSearchWeight(allWeight);
                std::cout << "___【 Load Weights 】: " << txt_path << " ___" << std::endl;

            }
        }
        else
        {
            std::cout << "___【 Warning! 】: " << txt_path << "does't exist! Degrade to use weavess::TYPE::ALL_WEIGHT ___" << std::endl;
            getFinalIndex()->setSearchWeight(allWeight);
        }
        auto e1 = std::chrono::high_resolution_clock::now();
        std::cout << "==============================================" << std::endl;
        std::cout << "__LOAD SEARCH-WEIGHTS FINISH__" << std::endl;
        std::chrono::duration<double> load_info_time = e1 - s1;
        std::cout << "^^^^^^ load_search_weight() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        if (getFinalIndex()->getSearchWeight().size()) {
            std::cout << "========== LOADED search-weight Information Summary ========================" << std::endl;
            std::cout << "getFinalIndex()->getSearchWeight(): " << getFinalIndex()->getSearchWeight().size() << ", " << getFinalIndex()->getSearchWeight()[0].size() << std::endl;
            std::cout << "search-weight for query 0: \n     ";
            for (int i = 0; i < getFinalIndex()->getSearchWeight()[0].size(); i++)
            {
                std::cout << getFinalIndex()->getSearchWeight()[0][i] << ", ";
            }
            std::cout << std::endl;
            std::cout << "==============================================" << std::endl;
            std::cout << smart_final_index_->getParam().toString() << std::endl;
            std::cout << "==============================================\n\n\n" << std::endl;
        }
        else 
        {
            std::cout << "========== NO loaded-search-weight ---> use AllWeights ========================" << std::endl;
            std::cout << "==============================================" << std::endl;
            std::cout << smart_final_index_->getParam().toString() << std::endl;
            std::cout << "==============================================\n\n\n" << std::endl;
        }
        

        return this;
    }

    /**
     * preliminary: 包括分field-groups，确认对应的L_refine，R_refine等
     */
    SmartIndexBuilder *SmartIndexBuilder::preliminary(Parameters &parameters, TYPE group_type, TYPE param_type)
    {
        auto sa = std::chrono::high_resolution_clock::now();

        auto *b = new ComponentPreliminary_smart(smart_final_index_);

        // ## 1. seperate field-groups
        auto s1 = std::chrono::high_resolution_clock::now();

        if (group_type == CLUSTER_GROUP)
        {
            b->SeperateInner_smart_ClusterGroup(parameters);
        }
        else if (group_type == GROUP_EQU_NO_TOTAL)
        {
            b->SeperateInner_smart_GroupEquNoTotal(parameters);
        }
        else if (group_type == GROUP_FUSION)
        {
            b->SeperateInner_smart_GroupFusion(parameters);
        }
        else if (group_type == GROUP_ALL_WEIGHT)
        {
            b->SeperateInner_smart_GroupAllWeight(parameters);
        }
        else
        {
            std::cerr << "[ERROR]: Don't have group_type: " << group_type << std::endl;
        }
        auto e1 = std::chrono::high_resolution_clock::now();
        // ## Show Group List information
        auto groupList = smart_final_index_->getGroupList();
        std::cout << "\n========== Group List Summary ========================" << std::endl;
        std::cout << "There are " << groupList.size() << " different field-groups." << std::endl;
        std::cout << "-----------------------" << std::endl;
        for (int g = 0; g < groupList.size(); g++)
        {
            std::cout << "- Group " << (g + 1) << ": ";
            for (int i = 0; i < groupList[g].size(); i++)
            {
                std::cout << groupList[g][i] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "==============================================" << std::endl;
        std::chrono::duration<double> load_info_time = e1 - s1;
        std::cout << "^^^^^^ preliminary->sepGroup() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        // ## 2. 准备每个index需要的parameters
        s1 = std::chrono::high_resolution_clock::now();
        if (param_type == PARAM_EQUAL)
        {
            b->PrepareParameter_smart_Equal(parameters);
        }
        else
        {
            std::cerr << "[ERROR]: Don't have param_type: " << param_type << std::endl;
        }
        e1 = std::chrono::high_resolution_clock::now();

        // ## Show Group List information
        auto param = smart_final_index_->getRRefineList();
        std::cout << "\n========== R, L parameter list Summary ========================" << std::endl;
        std::cout << "R_refine: " << std::endl;
        for (int g = 0; g < param.size(); g++)
        {
            std::cout << param[g] << ", ";
        }
        std::cout << "\n-----------------------" << std::endl;
        param = smart_final_index_->getLRefineList();
        std::cout << "L_refine: " << std::endl;
        for (int g = 0; g < param.size(); g++)
        {
            std::cout << param[g] << ", ";
        }
        std::cout << std::endl;
        std::cout << "==============================================" << std::endl;

        load_info_time = e1 - s1;
        std::cout << "^^^^^^ preliminary->sepPara() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "__PRELIMINARY FINISH__" << std::endl;
        load_info_time = e1 - sa;
        std::cout << "^^^^^^ preliminary() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        return this;
    }

    /** XMT, 2025.04.17
     * build init graph
     * @param type init type
     * @return pointer of builder
     */
    SmartIndexBuilder *SmartIndexBuilder::init(TYPE type, TYPE dist_type)
    {
        s = std::chrono::high_resolution_clock::now();
        ComponentInit_smart *a = nullptr;

        if (type == INIT_RAND)
        {
            std::cout << "__INIT(4SMART) : RAND__" << std::endl;
            a = new ComponentInitRand_smart(smart_final_index_);
        }
        else if (weavess::INIT_HNSW_FUSION)
        {
            std::cout << "__INIT(4SMART) : INIT_HNSW_FUSION__" << std::endl;
            a = new ComponentInitHNSW_Fusion(smart_final_index_);
        }
        else
        {
            std::cerr << "__INIT(4SMART) : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        a->InitInner_smart(dist_type);

        e = std::chrono::high_resolution_clock::now();

        std::cout << "==============================================" << std::endl;
        std::cout << "__INIT FINISH__" << std::endl;
        auto e1 = std::chrono::high_resolution_clock::now();

        if (type == INIT_RAND)
        {
            // ## Show InitGraph Summary
            std::vector<int> groupList;
            for (int i = 0; i < smart_final_index_->getGroupNum(); i++)
            {
                groupList.emplace_back(i);
            }
            smart_final_index_->summaryFinalGraph(groupList, {0, 1, smart_final_index_->getBaseLen() - 1});
        }

        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ init() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        return this;
    }

    /**
     * build refine graph
     * @param type refine type
     * @return
     */
    SmartIndexBuilder *SmartIndexBuilder::refine(TYPE type, TYPE dist_type)
    {
        ComponentRefine_smart *a = nullptr;

        if (type == INDEX_SMART)
        {
            std::cout << "__REFINE : SMART_INDEX__" << std::endl;
            a = new ComponentRefineSmart(smart_final_index_);
        }
        else if (type == INDEX_ORACLE_BASELINE)
        {
            std::cout << "__REFINE : BASELINE_ORACLE_INDEX__" << std::endl;
            std::cout << "__REFINE : ⚠️ 但是我感觉好像改了relaGroup之后直接用SmartIndex的部分就行__" << std::endl;
            a = new ComponentRefineSmart_Oracle(smart_final_index_);
            // a = new ComponentRefineSmart_CutGraph(smart_final_index_);
        }
        else if (type == INDEX_VAMANA_BASELINE)
        {
            std::cout << "__REFINE : BASELINE_VAMANA_TYPE_INDEX__" << std::endl;
            // a = new ComponentRefineSmart_BaselineVamana(smart_final_index_);
            std::cout << "__REFINE : 但是我感觉好像改了relaGroup之后直接用SmartIndex的部分就行__" << std::endl;
            a = new ComponentRefineSmart(smart_final_index_);
            // a = new ComponentRefineSmart_CutGraph(smart_final_index_);
        }
        else
        {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->RefineInner_smart(dist_type);

        std::cout << "===================" << std::endl;
        std::cout << "__REFINE ROUND " << getFinalIndex()->getRefineRound() << ": FINISH__" << std::endl;
        e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ this round refine() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;
        std::cout << "===================" << std::endl;
        showPresentTime();

        return this;
    }

    /**
     * graph connectivity enforcer
     * @param type connectivity check type
     * @return
     */
    SmartIndexBuilder *SmartIndexBuilder::connectivity_enforcer(TYPE type, TYPE dist_type)
    {
        ComponentConnectEnforcer_smart *a = nullptr;

        if (type == CONNECT_RELA)
        {
            std::cout << "__CONNECTIVITY : CONNECT_RELA__" << std::endl;
            a = new ComponentRelaConnectEnforcer(smart_final_index_);
        }
        else
        {
            std::cerr << "__REFINE : WRONG TYPE__" << std::endl;
        }

        a->ConnectEnforcerInner_smart(dist_type);

        std::cout << "===================" << std::endl;
        std::cout << "__CONNECTIVITY : FINISH__" << std::endl;
        std::cout << "===================" << std::endl;
        e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ connect() time is " << load_info_time.count() << " ^^^^^^ \n\n\n"
                  << std::endl;

        return this;
    }







    SmartIndexBuilder *SmartIndexBuilder::save_graph(TYPE type, char *graph_file)
    {
        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        if (type == INDEX_SMART)
        {
            smart_final_index_->loadEnabledGroupsCacheFromTxt(
                                "../include/python_file/cluster_query_weight_cache_2.txt"
                                );
            smart_final_index_->compactNewFinalGraphList();
            std::cout << "__ SAVE: INDEX_SMART ___" << std::endl;
            std::cout << "saving to graph file: " << graph_file << std::endl;
            int group_num = smart_final_index_->getGroupNum();
            unsigned len = smart_final_index_->getBaseLen();
            out.write((char *)&group_num, sizeof(int));
            out.write((char *)&len, sizeof(unsigned));
            // ## 分别存储每个group-index对应的信息：Save for each group-index
            for (int g = 0; g < group_num; g++)
            {
                // -- 1. Enter-point
                out.write((char *)&smart_final_index_->each_ep_[g], sizeof(unsigned));

                // -- 2. Group representative weight vector / Group Element
                if (smart_final_index_->getGroupRepreList().size())
                {
                    int placeholder = 0;
                    out.write((char *)&placeholder, sizeof(int));
                    out.write((char *)smart_final_index_->getGroupRepreList()[g].data(), smart_final_index_->getFieldNum() * sizeof(float));
                }
                else
                {
                    int numElement = smart_final_index_->getGroupList()[g].size();
                    out.write((char *)&numElement, sizeof(int));
                    out.write((char *)smart_final_index_->getGroupList()[g].data(), numElement * sizeof(int));
                }

                // -- 3. Related Edge Groups
                int numRela = smart_final_index_->getRelaCheckList()[g].size();
                out.write((char *)&numRela, sizeof(int));
                out.write((char *)smart_final_index_->getRelaCheck(g).data(), numRela * sizeof(int));

                int GK;
                for (unsigned i = 0; i < len; i++)
                {
                    auto &graph_i = smart_final_index_->getFinalGraph(g)[i];
                    GK = graph_i.size();
                    std::vector<unsigned> tmp;
                    for (unsigned j = 0; j < GK; j++)
                    {
                        tmp.push_back(graph_i[j].id);
                    }

                    out.write((char *)&GK, sizeof(int));
                    out.write((char *)tmp.data(), GK * sizeof(unsigned));
                }
            }

            // 4. 存correlated group 相关
            // 4.1 rela_thresh
            float rela_thresh = smart_final_index_->getRelaThresh();
            out.write((char *)&rela_thresh, sizeof(float));
            // 4.2 几个unique_workload_num， 及其对应的vector
            unsigned unique_workload_num = smart_final_index_->getUniqueWorkloadNum();
            out.write((char *)&unique_workload_num, sizeof(unsigned));
            // // cached weights: W x m
            // std::vector<float> cached_weights_flat_;          // size = W * m1
            out.write((char *)smart_final_index_->getUniqueWorkloadFlat().data(), unique_workload_num * smart_final_index_->getFieldNum() * sizeof(float));
            // 4.3 std::vector<uint32_t> cached_enabled_offsets_;    // size = W+1
            out.write((char *)smart_final_index_->getEnableGroupRecordOffset().data(), (unique_workload_num  + 1 ) * sizeof(uint32_t));
            // 4.4 std::vector<uint16_t> cached_enabled_flat_;    // enabled group （平铺）
            unsigned flat_len = smart_final_index_->getEnabledGroupFlat().size();
            out.write((char *)&flat_len, sizeof(unsigned));
            out.write((char *)smart_final_index_->getEnabledGroupFlat().data(), flat_len * sizeof(uint16_t));
            // -- 2026.01.24 [END] --

        }
        else if (type == INDEX_VAMANA_BASELINE)
        {
            smart_final_index_->compactNewFinalGraphList();
            std::cout << "__ SAVE: INDEX_VAMANA_BASELINE ___" << std::endl;
            std::cout << "saving to graph file: " << graph_file << std::endl;
            int group_num = smart_final_index_->getGroupNum();
            unsigned len = smart_final_index_->getBaseLen();
            out.write((char *)&group_num, sizeof(int));
            out.write((char *)&len, sizeof(unsigned));
            // ## 分别存储每个group-index对应的信息：Save for each group-index
            for (int g = 0; g < group_num; g++)
            {
                // -- 1. Enter-point
                out.write((char *)&smart_final_index_->each_ep_[g], sizeof(unsigned));

                // -- 2. Group representative weight vector / Group Element
                if (smart_final_index_->getGroupRepreList().size())
                {
                    int placeholder = 0;
                    out.write((char *)&placeholder, sizeof(int));
                    out.write((char *)smart_final_index_->getGroupRepreList()[g].data(), smart_final_index_->getFieldNum() * sizeof(float));
                }
                else
                {
                    int numElement = smart_final_index_->getGroupList()[g].size();
                    out.write((char *)&numElement, sizeof(int));
                    out.write((char *)smart_final_index_->getGroupList()[g].data(), numElement * sizeof(int));
                }

                // -- 3. Related Edge Groups
                int numRela = smart_final_index_->getRelaCheckList()[g].size();
                out.write((char *)&numRela, sizeof(int));
                out.write((char *)smart_final_index_->getRelaCheck(g).data(), numRela * sizeof(int));

                int GK;
                for (unsigned i = 0; i < len; i++)
                {
                    auto &graph_i = smart_final_index_->getFinalGraph(g)[i];
                    GK = graph_i.size();
                    std::vector<unsigned> tmp;
                    for (unsigned j = 0; j < GK; j++)
                    {
                        tmp.push_back(graph_i[j].id);
                    }

                    out.write((char *)&GK, sizeof(int));
                    out.write((char *)tmp.data(), GK * sizeof(unsigned));
                }
            }
        }
        else if (type == INDEX_HNSW_FUSION)
        {
            std::cout << "__ SAVE: INDEX_HNSW ___" << std::endl;
            unsigned enterpoint_id = smart_final_index_->enterpoint_->GetId();
            unsigned max_level = smart_final_index_->max_level_;
            out.write((char *)&enterpoint_id, sizeof(unsigned));
            out.write((char *)&max_level, sizeof(unsigned));
            for (unsigned i = 0; i < smart_final_index_->getBaseLen(); i++)
            {
                unsigned node_id = smart_final_index_->nodes_[i]->GetId();
                out.write((char *)&node_id, sizeof(unsigned));
                unsigned node_level = smart_final_index_->nodes_[i]->GetLevel() + 1;
                out.write((char *)&node_level, sizeof(unsigned));
                unsigned current_level_GK;
                for (unsigned j = 0; j < node_level; j++)
                {
                    current_level_GK = smart_final_index_->nodes_[i]->GetFriends(j).size();
                    out.write((char *)&current_level_GK, sizeof(unsigned));
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id = smart_final_index_->nodes_[i]->GetFriends(j)[k]->GetId();
                        out.write((char *)&current_level_neighbor_id, sizeof(unsigned));
                    }
                }
            }
            std::cout << "__ Graph has beed saved to file: " << graph_file << " __________________________" << std::endl;
            std::cout << "End of save_graph.\n --------------------------------------------------------" << std::endl;
            return this;
        }
        else
        {
            std::cerr << "[ERROR]: In save_graph(), don't have type: (at pos 1)" << type << std::endl;
            exit(-1);
        }
        out.close();
        std::cout << "__ FINISHED SAVE: INDEX_SMART ___" << std::endl;
        std::cout << "__ Graph has beed saved to file: " << graph_file << " __________________________" << std::endl;
        std::cout << "End of save_graph.\n --------------------------------------------------------" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        return this;
    }






    SmartIndexBuilder *SmartIndexBuilder::load_graph(TYPE type, char *graph_file)
    {
        std::cout << "___ LOAD_GRAPH FROM: " << graph_file << " ___" << std::endl;
        std::ifstream in(graph_file, std::ios::binary);
        auto s1 = std::chrono::high_resolution_clock::now();
        if (!in.is_open())
        {
            in.open("test/" + std::string(graph_file), std::ios::binary);
            if (!in.is_open())
            {
                std::cout << "graph_file path: \n   either: " << graph_file << std::endl;
                std::cout << "      or: " << graph_file << std::endl;
                std::cerr << "load graph error" << std::endl;
                exit(-1);
            }
        }

        if (type == INDEX_SMART)
        {
            std::cout << "___ LOAD: INDEX_SMART ___" << std::endl;

            int group_num;
            unsigned len;
            unsigned ep;
            std::vector<std::vector<int>> groupList, relaList;
            std::vector<std::vector<float>> groupRepreList;
            int numElement, numRela;
            int GK;

            in.read((char *)&group_num, sizeof(int));
            in.read((char *)&len, sizeof(unsigned));
            smart_final_index_->each_ep_.resize(group_num);
            smart_final_index_->getLoadGraphList().resize(group_num);

            // Reead for each group-index
            for (int g = 0; g < group_num; g++)
            {

                // -- 1. Enter-point of this edge group
                in.read((char *)&ep, sizeof(unsigned));
                smart_final_index_->each_ep_[g] = ep;

                // -- 2. Group representative weight vector / Group Element
                in.read((char *)&numElement, sizeof(int));
                if (numElement)
                {
                    std::vector<int> tmpGroup(numElement);
                    in.read((char *)tmpGroup.data(), numElement * sizeof(int));
                    groupList.emplace_back(tmpGroup);
                }
                else
                {
                    std::vector<float> tmpRepre(smart_final_index_->getFieldNum());
                    in.read((char *)tmpRepre.data(), smart_final_index_->getFieldNum() * sizeof(float));
                    groupRepreList.emplace_back(tmpRepre);
                    std::vector<int> tmpGroup;
                    std::cout << "-- tmpRepre: ";
                    for (int i = 0; i < smart_final_index_->getFieldNum(); i++)
                    {
                        std::cout << tmpRepre[i] << ", ";
                        if (tmpRepre[i])
                        {
                            tmpGroup.emplace_back(i);
                        }
                    }
                    std::cout << std::endl;
                    groupList.emplace_back(tmpGroup);
                }

                // -- 3. Related Edge Groups
                in.read((char *)&numRela, sizeof(int));
                std::vector<int> tmpRela(numRela);
                in.read((char *)tmpRela.data(), numRela * sizeof(int));
                relaList.emplace_back(tmpRela);
                std::cout << "-- tmpRela: ";
                for (int i = 0; i < numRela; i++) { std::cout << tmpRela[i] << ", "; }
                std::cout << std::endl;

                // Out-neighbors
                for (unsigned i = 0; i < len; i++)
                {
                    in.read((char *)&GK, sizeof(int));
                    std::vector<unsigned> tmpNN(GK);
                    in.read((char *)tmpNN.data(), GK * sizeof(unsigned));
                    smart_final_index_->getLoadGraph(g).push_back(tmpNN);
                }
            }
            // ## set up: related to group
            smart_final_index_->setGroupList(groupList);
            smart_final_index_->setGroupRepreList(groupRepreList);
            smart_final_index_->setRelaCheckList(relaList);


            // 4. 对应读取correlation相关部分
            // 4. 存correlated group 相关
            // 4.1 rela_thresh
            float rela_thresh;
            in.read((char *)&rela_thresh, sizeof(float));
            smart_final_index_->setRelaThresh(rela_thresh);
            // 4.2 几个unique_workload_num， 及其对应的vector
            unsigned unique_workload_num;
            in.read((char *)&unique_workload_num, sizeof(unsigned));
            smart_final_index_->setUniqueWorkloadNum(unique_workload_num); // W
            // -- std::vector<float> cached_weights_flat_;          // size = W * m1
            std::vector<float> weights_flat(smart_final_index_->getFieldNum() * unique_workload_num);
            in.read((char *)weights_flat.data(), smart_final_index_->getFieldNum() * unique_workload_num * sizeof(float));
            smart_final_index_->setUniqueWorkloadFlat(weights_flat);

            // 4.3 std::vector<uint32_t> cached_enabled_offsets_;    // size = W+1
            std::vector<uint32_t> enabled_offsets((unique_workload_num + 1 ));
            in.read((char *)enabled_offsets.data(), (unique_workload_num + 1 ) * sizeof(uint32_t));
            smart_final_index_->setEnableGroupRecordOffset(enabled_offsets);

            // 4.4 std::vector<uint16_t> cached_enabled_flat_;    // enabled group （平铺）
            unsigned flat_len;
            in.read((char *)&flat_len, sizeof(unsigned));
            std::vector<uint16_t> enabled_flat(flat_len);
            in.read((char *)enabled_flat.data(), flat_len * sizeof(uint16_t));
            smart_final_index_->setEnabledGroupFlat(enabled_flat);

            smart_final_index_->showSavedEnabledSummary();
        }
        else if (type == INDEX_VAMANA_BASELINE)
        {
            std::cout << "___ LOAD: INDEX_VAMANA_BASELINE ___" << std::endl;

            int group_num;
            unsigned len;
            unsigned ep;
            std::vector<std::vector<int>> groupList, relaList;
            std::vector<std::vector<float>> groupRepreList;
            int numElement, numRela;
            int GK;

            in.read((char *)&group_num, sizeof(int));
            in.read((char *)&len, sizeof(unsigned));
            smart_final_index_->each_ep_.resize(group_num);
            smart_final_index_->getLoadGraphList().resize(group_num);

            // Reead for each group-index
            for (int g = 0; g < group_num; g++)
            {
                // -- 1. Enter-point of this edge group
                in.read((char *)&ep, sizeof(unsigned));
                smart_final_index_->each_ep_[g] = ep;

                // -- 2. Group representative weight vector / Group Element
                in.read((char *)&numElement, sizeof(int));
                if (numElement)
                {
                    std::vector<int> tmpGroup(numElement);
                    in.read((char *)tmpGroup.data(), numElement * sizeof(int));
                    groupList.emplace_back(tmpGroup);
                }
                else
                {
                    std::vector<float> tmpRepre(smart_final_index_->getFieldNum());
                    in.read((char *)tmpRepre.data(), smart_final_index_->getFieldNum() * sizeof(float));
                    groupRepreList.emplace_back(tmpRepre);
                    std::vector<int> tmpGroup;
                    std::cout << "-- tmpRepre: ";
                    for (int i = 0; i < smart_final_index_->getFieldNum(); i++)
                    {
                        std::cout << tmpRepre[i] << ", ";
                        if (tmpRepre[i])
                        {
                            tmpGroup.emplace_back(i);
                        }
                    }
                    std::cout << std::endl;
                    groupList.emplace_back(tmpGroup);
                }

                // -- 3. Related Edge Groups
                in.read((char *)&numRela, sizeof(int));
                std::vector<int> tmpRela(numRela);
                in.read((char *)tmpRela.data(), numRela * sizeof(int));
                relaList.emplace_back(tmpRela);
                std::cout << "-- tmpRela: ";
                for (int i = 0; i < numRela; i++) { std::cout << tmpRela[i] << ", "; }
                std::cout << std::endl;

                // Out-neighbors
                for (unsigned i = 0; i < len; i++)
                {
                    in.read((char *)&GK, sizeof(int));
                    std::vector<unsigned> tmpNN(GK);
                    in.read((char *)tmpNN.data(), GK * sizeof(unsigned));
                    smart_final_index_->getLoadGraph(g).push_back(tmpNN);
                }
            }
            // ## set up: related to group
            smart_final_index_->setGroupList(groupList);
            smart_final_index_->setGroupRepreList(groupRepreList);
            smart_final_index_->setRelaCheckList(relaList);
        }
        else if (type == INDEX_HNSW_FUSION)
        {
            smart_final_index_->nodes_.resize(smart_final_index_->getBaseLen());
            for (unsigned i = 0; i < smart_final_index_->getBaseLen(); i++)
            {
                smart_final_index_->nodes_[i] = new weavess::HNSW::HnswNode(0, 0, 0, 0);
            }
            unsigned enterpoint_id;
            in.read((char *)&enterpoint_id, sizeof(unsigned));
            in.read((char *)&smart_final_index_->max_level_, sizeof(unsigned));

            for (unsigned i = 0; i < smart_final_index_->getBaseLen(); i++)
            {
                unsigned node_id, node_level, current_level_GK;
                in.read((char *)&node_id, sizeof(unsigned));
                smart_final_index_->nodes_[node_id]->SetId(node_id);
                in.read((char *)&node_level, sizeof(unsigned));
                smart_final_index_->nodes_[node_id]->SetLevel(node_level);
                for (unsigned j = 0; j < node_level; j++)
                {
                    in.read((char *)&current_level_GK, sizeof(unsigned));
                    std::vector<weavess::HNSW::HnswNode *> tmp;
                    for (unsigned k = 0; k < current_level_GK; k++)
                    {
                        unsigned current_level_neighbor_id;
                        in.read((char *)&current_level_neighbor_id, sizeof(unsigned));
                        tmp.push_back(smart_final_index_->nodes_[current_level_neighbor_id]);
                    }
                    smart_final_index_->nodes_[node_id]->SetFriends(j, tmp);
                }
            }
            smart_final_index_->enterpoint_ = smart_final_index_->nodes_[enterpoint_id];
            auto e1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e1 - s1;
            std::cout << "^^^^^^ individual load_graph() time is : " << diff.count() << " ^^^^^^ \n\n\n";
            return this;
        }
        else
        {
            std::cerr << "[ERROR]: In load_graph(), don't have type (at pos 2): " << type << std::endl;
            exit(-1);
        }

        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e1 - s1;
        std::cout << "^^^^^^ individual load_graph() time is : " << diff.count() << " ^^^^^^ \n\n\n"
                  << std::endl;
        return this;
    }






    /**
     * offline search
     * @param entry_type
     * @param route_type
     * @return
     */
    SmartIndexBuilder *SmartIndexBuilder::search(TYPE entry_type, TYPE route_type, TYPE L_type, TYPE weight_type, TYPE dist_type)
    {
        std::cout << "__SEARCH__" << std::endl;

        unsigned K = 10;
        if (smart_final_index_->getParam().exist("K_search") == false)
        {
            std::cout << "#### K_search(ka-ann) no exist. I am using the default K_search = 10..." << std::endl;
            smart_final_index_->getParam().set<unsigned>("K_search", K);
        }
        else
        {
            std::cout << "#### K_search(ka-ann) = " << smart_final_index_->getParam().get<unsigned>("K_search") << std::endl;
            K = smart_final_index_->getParam().get<unsigned>("K_search");
        }

        // GROUND TRUTH
        ComponentGroundTruth_smart *g = new ComponentGroundTruth_smart(smart_final_index_);

        std::vector<std::vector<unsigned>> res;
        std::vector<std::vector<float>> dist_res;

        // ENTRY
        ComponentSearchEntry_smart *a = nullptr;
        if (entry_type == SEARCH_ENTRY_CENTROID)
        {
            std::cout << "__SEARCH ENTRY : CENTROID__" << std::endl;
            a = new ComponentSearchEntryCentroid_smart(smart_final_index_);
        }
        else if (entry_type == SEARCH_ENTRY_NONE_FUSION)
        {
            std::cout << "__SEARCH ENTRY : NONE_FUSION__" << std::endl;
            a = new ComponentSearchEntryNone_Fusion(smart_final_index_);
        }
        else
        {
            std::cerr << "__SEARCH ENTRY : WRONG TYPE__" << std::endl;
            std::cerr << "entry_type: " << entry_type << std::endl;
            exit(-1);
        }


        // ROUTE
        ComponentSearchRoute_smart *b = nullptr;
        if (route_type == ROUTER_GREEDY)
        {
            std::cout << "__ROUTER : GREEDY__" << std::endl;
            std::cout << "route_type  = " << route_type << std::endl;
            b = new ComponentSearchRouteGreedy_smart(smart_final_index_);
        }
        else if (route_type == ROUTER_HNSW_FUSION)
        {
            std::cout << "__ROUTER : HNSW_FUSION__" << std::endl;
            b = new ComponentSearchRouteHNSW_Fusion(smart_final_index_);
        }
        else
        {
            std::cerr << "__ROUTER : WRONG TYPE__" << std::endl;
            std::cout << "route_type = " << route_type << ",  ROUTER_GREEDY = " <<  ROUTER_GREEDY << std::endl;
            exit(-1);
        }


        ComponentSearchFlowLoadWeight_smart *c = new ComponentSearchFlowLoadWeight_smart(smart_final_index_);

        std::vector<std::vector<float>> recall_matrix, latency_matrix, hop_matrix, distCount_matrix, usedDimCount_matrix, DCHTCount_matrix;
        if (weight_type == LOAD_WEIGHT)
        {
            std::cout << "__SEARCH FLOW(using weight) : LOAD WEIGHT(ONCE)__" << std::endl;
            std::cout << "__GROUND TRUTH : LOAD WEIGHT(ONCE)__" << std::endl;
            std::cout << "__with (qi, wi) of size: " << smart_final_index_->getSearchWeight().size() << " .... (assert |SearchWorkload| = " << smart_final_index_->getSearchWeight().size() <<  " == " << smart_final_index_->getQueryLen() << " = numQuery __" << std::endl;
            assert(smart_final_index_->getSearchWeight().size() == smart_final_index_->getQueryLen());
            g->GroundInner_smart(K, dist_type);

            if (L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT__" << std::endl;
                std::cout << "L_search = K = " << K << std::endl;
                unsigned L = K;
                smart_final_index_->getParam().set<unsigned>("L_search", K);
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                c->FlowInner_WeightOnce_smart_FallbackIntersect_forDiff_wi(K, L, a, b, recall_matrix[0][0], latency_matrix[0][0], hop_matrix[0][0], distCount_matrix[0][0], dist_type);
            }
            else if (L_type == L_SEARCH_ASSIGN_INTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_INTERSECT__" << std::endl;
                std::cout << "L_search = K = " << K << std::endl;
                unsigned L = K;
                smart_final_index_->getParam().set<unsigned>("L_search", K);
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                c->FlowInner_WeightOnce_smart_intersect(K, L, a, b, recall_matrix[0][0], latency_matrix[0][0], hop_matrix[0][0], distCount_matrix[0][0], dist_type);
            }
            else if (L_type == L_SEARCH_ASSIGN_ALL_INDEX)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_ALL_INDEX__" << std::endl;
                std::cout << "L_search = K = " << K << std::endl;
                unsigned L = K;
                smart_final_index_->getParam().set<unsigned>("L_search", K);
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(1));
                c->FlowInner_WeightOnce_smart_allIndex(K, L, a, b, recall_matrix[0][0], latency_matrix[0][0], hop_matrix[0][0], distCount_matrix[0][0], dist_type);
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT" << std::endl;
                std::vector<unsigned> LRate;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                c->FlowInner_WeightOnceControL_smart_FallbackIntersect_forDiff_wi(K, LRate, a, b, recall_matrix[0], latency_matrix[0], hop_matrix[0], distCount_matrix[0], dist_type);
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_INTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_INTERSECT__" << std::endl;
                std::vector<unsigned> LRate;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                c->FlowInner_WeightOnceControL_smart_intersect(K, LRate, a, b, recall_matrix[0], latency_matrix[0], hop_matrix[0], distCount_matrix[0], dist_type);
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX" << std::endl;
                std::vector<unsigned> LRate;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(1, std::vector<float>(LRate.size()));
                c->FlowInner_WeightOnceControL_smart_allIndex(K, LRate, a, b, recall_matrix[0], latency_matrix[0], hop_matrix[0], distCount_matrix[0], dist_type);
            }
            else
            {
                std::cerr << "__L_type : WRONG TYPE (pos 1): " << L_type << "__" << std::endl;
                exit(-1);
            }
        }
        else if (weight_type == ALL_WEIGHT)
        {
            std::cout << "__SEARCH FLOW(using weight) : ALL_WEIGHT(2^m - 1)__" << std::endl;
            std::vector<unsigned> LRate;
            std::vector<float> weight(smart_final_index_->getFieldNum());
            unsigned numComb = ((1 << smart_final_index_->getFieldNum()) - 1);
            std::chrono::duration<double> addition_diff{0};

            if (L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT__" << std::endl;
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
            }
            else if (L_type == L_SEARCH_ASSIGN_INTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_INTERSECT__" << std::endl;
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                { 
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                { 
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_INTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_EXACT_REPRE)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_EXACT_REPRE__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else
            {
                std::cerr << "__L_type : WRONG TYPE (pos 3): " << L_type << "__" << std::endl;
                exit(-1);
            }

            // ### 生成所有可能的weight组合，对每种weight进行search
            for (int w = 1; w < (1 << smart_final_index_->getFieldNum()); ++w)
            {   
                for (size_t j = 0; j < smart_final_index_->getFieldNum(); ++j)
                {
                    weight[j] = (w & (1 << j)) ? 1.0f : 0.0f; // 第 j 位
                }
                if (true)
                {
                    // 打印当前组合
                    std::cout << "Combination " << w << " / " << ((1 << smart_final_index_->getFieldNum()) - 1) << " : ";
                    for (float w : weight)
                    {
                        std::cout << w << " ";
                    }
                    std::cout << std::endl;
                }

                std::vector<std::vector<float>> search_weight(smart_final_index_->getQueryLen(), weight);
                smart_final_index_->setSearchWeight(search_weight);

                std::cout << "__GROUND TRUTH : ALL_WEIGHT " << w << " / " << ((1 << smart_final_index_->getFieldNum()) - 1) << " __" << std::endl;
                g->GroundInner_smart_load(w, K, dist_type);

                if (L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT)
                {
                    std::cout << "__L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT__" << std::endl;
                    std::cout << "L_search = K = " << K << std::endl;
                    unsigned L = K;
                    c->FlowInner_WeightOnce_smart_FallbackIntersect(K, L, a, b, recall_matrix[w - 1][0], latency_matrix[w - 1][0], hop_matrix[w - 1][0], distCount_matrix[w - 1][0], dist_type);
                }
                else if (L_type == L_SEARCH_ASSIGN_INTERSECT)
                {
                    std::cout << "__L_type == L_SEARCH_ASSIGN_INTERSECT__" << std::endl;
                    std::cout << "L_search = K = " << K << std::endl;
                    unsigned L = K;
                    c->FlowInner_WeightOnce_smart_intersect(K, L, a, b, recall_matrix[w - 1][0], latency_matrix[w - 1][0], hop_matrix[w - 1][0], distCount_matrix[w - 1][0], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_FallbackIntersect(K, LRate, a, b, recall_matrix[w - 1], latency_matrix[w - 1], hop_matrix[w - 1], distCount_matrix[w - 1], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_allIndex(K, LRate, a, b, recall_matrix[w - 1], latency_matrix[w - 1], hop_matrix[w - 1], distCount_matrix[w - 1], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_INTERSECT)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_INTERSECT__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_intersect(K, LRate, a, b, recall_matrix[w - 1], latency_matrix[w - 1], hop_matrix[w - 1], distCount_matrix[w - 1], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_EXACT_REPRE)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_EXACT_REPRE__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_exactRepre(K, LRate, a, b, recall_matrix[w - 1], latency_matrix[w - 1], hop_matrix[w - 1], distCount_matrix[w - 1], dist_type);
                }
                else
                {
                    std::cerr << "__L_type : WRONG TYPE__" << std::endl;
                    exit(-1);
                }
                std::cout << "\n\n\n"
                          << std::endl;
            }
        }
        else if (weight_type == LOADED_ALL_WEIGHT)
        {
            std::vector<unsigned> LRate;
            std::vector<float> weight(smart_final_index_->getFieldNum());
            std::vector<std::vector<float>> allWeight = smart_final_index_->getSearchWeight();
            unsigned numComb = allWeight.size();
            std::chrono::duration<double> addition_diff{0};
            int field_num = smart_final_index_->getFieldNum();

            if (allWeight.size() == 0)
            {
                std::cout << "___【 Warning! 】: No weight provided! Degrade to use weavess::TYPE::ALL_WEIGHT ___" << std::endl;
                search(entry_type, route_type, L_type, weavess::TYPE::ALL_WEIGHT, dist_type);
                return this;
            }

            std::cout << "__SEARCH FLOW(using weight) : LOADED_ALL_WEIGHT of " << numComb << " different weights __" << std::endl;
            if (L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT__" << std::endl;
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
            }
            else if (L_type == L_SEARCH_ASSIGN_INTERSECT)
            {
                std::cout << "__L_type == L_SEARCH_ASSIGN_INTERSECT__" << std::endl;
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(1));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else if (L_type == L_RECALL_SEARCH_CONTROL_INTERSECT)
            {
                std::cout << "__L_type == L_RECALL_SEARCH_CONTROL__" << std::endl;
                if (route_type == weavess::TYPE::ROUTER_HNSW_FUSION)
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                else
                {
                    LRate = {1, 2, 3, 5, 10, 20, 50, 100, 200};
                }
                recall_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                latency_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                hop_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
                distCount_matrix = std::vector<std::vector<float>>(numComb, std::vector<float>(LRate.size()));
            }
            else
            {
                std::cerr << "__L_type : WRONG TYPE (pos 4): " << L_type << "__" << std::endl;
                exit(-1);
            }

            // ### 所有可能的weight组合，对每种weight进行search
            for (unsigned w = 0; w < numComb; ++w)
            {
                for (size_t j = 0; j < smart_final_index_->getFieldNum(); ++j)
                {
                    weight[j] = allWeight[w][j]; // 第 j 位
                }
                if (true)
                {
                    std::cout << "Combination " << (w + 1) << " / " << numComb << " : ";
                    for (float w : weight)
                    {
                        std::cout << w << " ";
                    }
                    std::cout << std::endl;
                }
                // -- 更改：查询&ground-truth时候用的所有search_weight
                std::cout << "__GROUND TRUTH : ALL_WEIGHT " << (w + 1) << " / " << numComb << " : ";
                std::vector<std::vector<float>> search_weight(smart_final_index_->getQueryLen(), weight); 
                smart_final_index_->setSearchWeight(search_weight);
                g->GroundInner_smart(K, dist_type);

                if (L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT)
                {
                    std::cout << "__L_type == L_SEARCH_ASSIGN_FALLBACKINTERSECT__" << std::endl;
                    std::cout << "L_search = K = " << K << std::endl;
                    unsigned L = K;
                    c->FlowInner_WeightOnce_smart_FallbackIntersect(K, L, a, b, recall_matrix[w][0], latency_matrix[w][0], hop_matrix[w][0], distCount_matrix[w][0], dist_type);
                }
                else if (L_type == L_SEARCH_ASSIGN_INTERSECT)
                {
                    std::cout << "__L_type == L_SEARCH_ASSIGN_INTERSECT__" << std::endl;
                    std::cout << "L_search = K = " << K << std::endl;
                    unsigned L = K;
                    c->FlowInner_WeightOnce_smart_intersect(K, L, a, b, recall_matrix[w][0], latency_matrix[w][0], hop_matrix[w][0], distCount_matrix[w][0], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_FallbackIntersect(K, LRate, a, b, recall_matrix[w], latency_matrix[w], hop_matrix[w], distCount_matrix[w], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_ALL_INDEX__" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_allIndex(K, LRate, a, b, recall_matrix[w], latency_matrix[w], hop_matrix[w], distCount_matrix[w], dist_type);
                }
                else if (L_type == L_RECALL_SEARCH_CONTROL_INTERSECT)
                {
                    std::cout << "__L_type == L_RECALL_SEARCH_CONTROL_INTERSECT at this pos 3" << std::endl;
                    c->FlowInner_WeightOnceControL_smart_intersect(K, LRate, a, b, recall_matrix[w], latency_matrix[w], hop_matrix[w], distCount_matrix[w], dist_type);
                }
                else
                {
                    std::cerr << "__L_type : WRONG TYPE__" << std::endl;
                    exit(-1);
                }
                std::cout << "\n\n\n"
                          << std::endl;

            }
        }
        else
        {
            std::cerr << "__SEARCH FLOW : WRONG TYPE__" << std::endl;
            exit(-1);
        }

        // ## Show recall-serch result
        std::cout << "\n\n\n######### Recall-Serch Result ###########" << std::endl;
        std::vector<float> aver_recall(recall_matrix[0].size(), 0.0), aver_latency(recall_matrix[0].size(), 0.0);
        std::vector<float> aver_hop(hop_matrix[0].size(), 0.0), aver_distCount(distCount_matrix[0].size(), 0.0);
        std::vector<float> aver_usedDimPerObject(hop_matrix[0].size(), 0.0), aver_usedDim(hop_matrix[0].size(), 0.0), aver_DCHTCount(hop_matrix[0].size(), 0.0);
        unsigned numWeight = recall_matrix.size();
        {
            for (unsigned w = 0; w < recall_matrix.size(); w++)
            {
                std::cout << "Performance with weight choice " << w + 1 << std::endl;
                ;
                std::cout << K << "-NN recall-latency results:" << std::endl;
                for (unsigned i = 0; i < recall_matrix[w].size(); i++)
                {
                    std::cout << recall_matrix[w][i] << ", " << latency_matrix[w][i] << std::endl;
                    aver_recall[i] += recall_matrix[w][i] / numWeight;
                    aver_latency[i] += latency_matrix[w][i] / numWeight;
                }
                std::cout << K << "-NN HopCount/query, DistCount/query:" << std::endl;
                for (unsigned i = 0; i < hop_matrix[w].size(); i++)
                {
                    float hop = hop_matrix[w][i] / smart_final_index_->getQueryLen();
                    float distCount = distCount_matrix[w][i] / smart_final_index_->getQueryLen();
                    std::cout << hop << ", " << distCount << std::endl;
                    aver_hop[i] += hop / numWeight;
                    aver_distCount[i] += distCount / numWeight;
                }
                if (usedDimCount_matrix.size()) {
                    std::cout << K << "-NN usedDimension/(query*DistCount), usedDimension/query, DCHTCount/query:" << std::endl;
                    for (unsigned i = 0; i < usedDimCount_matrix[w].size(); i++)
                    {
                        float distCount = distCount_matrix[w][i] / smart_final_index_->getQueryLen();
                        float usedDimPerObject = usedDimCount_matrix[w][i] / distCount;
                        std::cout << usedDimPerObject << ", " << usedDimCount_matrix[w][i] << ", " << DCHTCount_matrix[w][i] << std::endl;
                        aver_usedDimPerObject[i] += usedDimPerObject / numWeight;
                        aver_usedDim[i] += usedDimCount_matrix[w][i] / numWeight;
                        aver_DCHTCount[i] += DCHTCount_matrix[w][i] / numWeight;
                    }
                }
                std::cout << "----------------\n"
                          << std::endl;
            }
        }

        std::cout << "######### Final Summary #########" << std::endl;
        std::cout << "- Average recall-latency performance (with different weight, but same L):" << std::endl;
        for (unsigned i = 0; i < aver_recall.size(); i++)
        {
            std::cout << aver_recall[i] << ", " << aver_latency[i] << std::endl;
        }
        std::cout << "- Average HopCount/query, DistCount/query performance (with different weight, but same L):" << std::endl;
        for (unsigned i = 0; i < aver_hop.size(); i++)
        {
            std::cout << aver_hop[i] << ", " << aver_distCount[i] << std::endl;
        }
        if (usedDimCount_matrix.size()) {
            std::cout << "- Average usedDimension/(query*DistCount), usedDimension/query, DCHTCount/query performance (with different weight, but same L):" << std::endl;
            for (unsigned i = 0; i < aver_usedDim.size(); i++)
            {
                std::cout << aver_usedDimPerObject[i] << ", " << aver_usedDim[i] << ", " << aver_DCHTCount[i] << std::endl;
            }
        }
        std::cout << "#############################################\n\n" << std::endl;

        e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }

}
