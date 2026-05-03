#ifndef INDEXCODEFROMMAIN_H
#define INDEXCODEFROMMAIN_H

#include <iostream>


#include <builder.h>
#include <exp_data.h>

// std::vector<unsigned> k_args({5, 10, 20, 50, 100});



void PolyGraph(xmt::Parameters &parameters) {
    std::string dist_type_name = parameters.get<std::string>("dist_type");
    xmt::TYPE dist_type;
    std::chrono::duration<double> load_time(0.0);
    std::chrono::duration<double> update_time(0.0);
    parameters.set<float>("alpha2", 1.0);

    if (dist_type_name == "euclidean") {
        dist_type = xmt::TYPE::DIST_EUCLIDEAN;
    } else if (dist_type_name == "cosDist") {
        dist_type = xmt::TYPE::DIST_COS;
    } else if (dist_type_name == "cosSim") {
        dist_type = xmt::TYPE::DIST_COS_SIMILARITY;
    } else {
        std::cout << "error here about distance method..." << std::endl;
        exit(-1);
    }

    auto s_all = std::chrono::high_resolution_clock::now();

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string graph_file = parameters.get<std::string>("graph_file");

    if (dist_type_name == "euclidean") {
        dist_type = xmt::TYPE::DIST_EUCLIDEAN;
    } else if (dist_type_name == "cosDist") {
        dist_type = xmt::TYPE::DIST_COS;
    } else if (dist_type_name == "cosSim") {
        dist_type = xmt::TYPE::DIST_COS_SIMILARITY;
    } else {
        std::cout << "error here about distance method..." << std::endl;
        exit(-1);
    }
    
    auto *smart_builder = new xmt::MultiIndexBuilder(num_threads);

    if ( parameters.get<std::string>("exc_type") == "build" ) 
    {   // build
        smart_builder -> load(parameters);

        smart_builder -> preliminary(parameters, xmt::CLUSTER_GROUP, xmt::PARAM_EQUAL);

        smart_builder -> init(xmt::INIT_RAND, dist_type);
        std::cout << "Init cost: " << smart_builder->GetBuildTime().count() << std::endl;

        // ## 除非woAll，否则，每次refine后都接CE
        for (unsigned r = 0; r < 2; r++) {
            smart_builder -> refine(xmt::INDEX_SMART, dist_type); 
            if (parameters.get<std::string>("exc_type") != "build_woAll") {
                smart_builder -> connectivity_enforcer(xmt::CONNECT_RELA, dist_type);
            }
        }
        
        //  -- Construction finished; Show summary.
        std::cout << "\n===================" << std::endl;
        std::cout << "__ALL REFINE: FINISH__" << std::endl;
        std::cout << "save to graph file: " <<graph_file << std::endl;
        smart_builder -> save_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        std::cout << "Build cost: " << smart_builder->GetBuildTime().count() << std::endl;
        std::cout << "===================\n" << std::endl;
        
        // ## Show FinalGraph Summary
        std::vector<int> groupList;
        for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
        smart_builder->getFinalIndex()->summaryFinalGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "search") {   // search (FallbackIntersect) (one L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_SEARCH_ASSIGN_FALLBACKINTERSECT, xmt::TYPE::LOAD_WEIGHT,
                                dist_type);
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "recall_search") {   // recall_search_FallbackIntersect (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT, xmt::TYPE::LOAD_WEIGHT,
                                dist_type);
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("exc_type") == "all_recall_search") {   // all_recall_search_querySelectRepre_FallbackIntersect (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_FALLBACKINTERSECT, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type);
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("exc_type") == "all_recall_search_intersect") {   // all_recall_search_intersect (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_INTERSECT, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type);
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("exc_type") == "all_recall_search_allIndex") {   // all_recall_search_allIndex (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_SMART, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_ALL_INDEX, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type);
    }
    else {
        std::cout << "Alg name error: " << parameters.get<std::string>("alg") << std::endl;
        std::cout << "  or  " << std::endl;
        std::cout << "Exc_type name error: " << parameters.get<std::string>("exc_type") << std::endl;
        exit(-1);
    }

    auto e_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> run_time = e_all - s_all;
    std::cout << "^^^^^^^ Whole program running time (contains deleting) is: " << run_time.count() << std::endl;
};




/** 2025.06.20
 * vamana_baseline_index_group:
 *      vamana_equSpace; vamana_equNoTotal; vamana_fusion 都用这个
 */
void vamana_baseline_index_group(xmt::Parameters &parameters) {
    parameters.set<float>("alpha2", 1.2);

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string graph_file = parameters.get<std::string>("graph_file");
    std::string dataset = parameters.get<std::string>("dataset");
    std::string dist_type_name = parameters.get<std::string>("dist_type");
    xmt::TYPE dist_type;
    std::chrono::duration<double> load_time(0.0);
    std::chrono::duration<double> update_time(0.0);

    if (dist_type_name == "euclidean") {
        dist_type = xmt::TYPE::DIST_EUCLIDEAN;
    } else if (dist_type_name == "cosDist") {
        dist_type = xmt::TYPE::DIST_COS;
    } else if (dist_type_name == "cosSim") {
        dist_type = xmt::TYPE::DIST_COS_SIMILARITY;
    } else {
        std::cout << "error here about distance method..." << std::endl;
        exit(-1);
    }

    auto s_all = std::chrono::high_resolution_clock::now();
    
    auto *smart_builder = new xmt::MultiIndexBuilder(num_threads);

    if ( parameters.get<std::string>("exc_type") == "build") 
    {   // build
        smart_builder -> load(parameters);

        if (parameters.get<std::string>("alg") == "vamana_equNoTotal") {
            smart_builder -> preliminary(parameters, xmt::GROUP_EQU_NO_TOTAL, xmt::PARAM_EQUAL);
        } 
        else if (parameters.get<std::string>("alg") == "vamana_fusion") {
            smart_builder -> preliminary(parameters, xmt::GROUP_FUSION, xmt::PARAM_EQUAL);
        }
        else if (parameters.get<std::string>("alg") == "vamana_allWeight" || parameters.get<std::string>("alg") == "vamana_oracle") {
            smart_builder -> preliminary(parameters, xmt::GROUP_ALL_WEIGHT, xmt::PARAM_EQUAL);
        }
        else {
            std::cout << "Alg name error: " << parameters.get<std::string>("alg") << std::endl;
            exit(-1);
        }

        smart_builder -> init(xmt::INIT_RAND, dist_type);
        std::cout << "Init cost: " << smart_builder->GetBuildTime().count() << std::endl;

        // ## Refine
        if (parameters.get<std::string>("alg") == "vamana_oracle") 
        {
            for (unsigned r = 0; r < 2; r++) {
                smart_builder -> refine(xmt::INDEX_ORACLE_BASELINE,  dist_type);
            }
        }
        else 
        {
            for (unsigned r = 0; r < 2; r++) {
                smart_builder -> refine(xmt::INDEX_VAMANA_BASELINE,  dist_type);
            }
        }

        std::cout << "save to graph file: " <<graph_file << std::endl;
        smart_builder -> save_graph(xmt::TYPE::INDEX_VAMANA_BASELINE, &graph_file[0]);
        std::cout << "Build cost: " << smart_builder->GetBuildTime().count() << std::endl;

        // ## Show FinalGraph Summary
        std::vector<int> groupList;
        for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
        smart_builder->getFinalIndex()->summaryFinalGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "search") {   // search
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_VAMANA_BASELINE, &graph_file[0]);

        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_SEARCH_ASSIGN_INTERSECT, xmt::TYPE::LOAD_WEIGHT,
                                dist_type);
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "recall_search") {   // recall_search (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_VAMANA_BASELINE, &graph_file[0]);
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_INTERSECT, xmt::TYPE::LOAD_WEIGHT,
                                dist_type);
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("alg") == "vamana_oracle" && parameters.get<std::string>("exc_type") == "all_recall_search") {   // all_recall_search_exact_repre (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_VAMANA_BASELINE, &graph_file[0]);
        
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_EXACT_REPRE, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type);
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("exc_type") == "all_recall_search") {   // all_recall_search_intersect (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_VAMANA_BASELINE, &graph_file[0]); 
        {
            // -- Show summary of the temporary graph
            std::vector<int> groupList;
            for (int i = 0; i < smart_builder->getFinalIndex()->getGroupNum(); i++) { groupList.emplace_back(i); }
            smart_builder->getFinalIndex()->summaryLoadGraph(groupList, {0, 1, smart_builder->getFinalIndex()->getBaseLen()-1});
        }

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_CENTROID, xmt::TYPE::ROUTER_GREEDY, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_INTERSECT, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type);
    }
    else {
        std::cout << "Alg name error: " << parameters.get<std::string>("alg") << std::endl;
        std::cout << "  or  " << std::endl;
        std::cout << "Exc_type name error: " << parameters.get<std::string>("exc_type") << std::endl;
        exit(-1);
    }
    

    auto e_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> run_time = e_all - s_all;
    std::cout << "^^^^^^^ Whole program running time (contains deleting) is: " << run_time.count() << std::endl;
};











/** 2025.06.11
 * 使用MultiIndexBuilder版本的HNSW_FUSION
 */
void HNSW_FUSION(xmt::Parameters &parameters) {
    std::string dataset = parameters.get<std::string>("dataset");
    std::string dist_type_name = parameters.get<std::string>("dist_type");
    xmt::TYPE dist_type;

    std::chrono::duration<double> load_time(0.0);
    std::chrono::duration<double> update_time(0.0);

    if (dist_type_name == "euclidean") {
        dist_type = xmt::TYPE::DIST_EUCLIDEAN;
    } else if (dist_type_name == "cosDist") {
        dist_type = xmt::TYPE::DIST_COS;
    } else if (dist_type_name == "cosSim") {
        dist_type = xmt::TYPE::DIST_COS_SIMILARITY;
    } else {
        std::cout << "error here about distance method..." << std::endl;
        exit(-1);
    }

    auto s_all = std::chrono::high_resolution_clock::now();

    parameters.set<std::string>("base_path", "../dataset/concate/Concate_base.fvecs");
    parameters.set<std::string>("query_path", "../dataset/concate/Concate_query.fvecs");

    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string graph_file = parameters.get<std::string>("graph_file");

    auto *smart_builder = new xmt::MultiIndexBuilder(num_threads);


    if (parameters.get<std::string>("exc_type") == "build") {   // build
        smart_builder -> load(parameters);
        smart_builder -> init(xmt::INIT_HNSW_FUSION, dist_type);
        smart_builder -> save_graph(xmt::TYPE::INDEX_HNSW_FUSION, &graph_file[0]);

        std::cout << "Build cost: " << smart_builder->GetBuildTime().count() << std::endl;
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "search") {    // search
        smart_builder -> load(parameters);
        smart_builder -> load_graph(xmt::TYPE::INDEX_HNSW_FUSION, &graph_file[0]);

        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_NONE_FUSION, xmt::TYPE::ROUTER_HNSW_FUSION,
                                xmt::TYPE::L_SEARCH_ASSIGN_ALL_INDEX, xmt::TYPE::LOAD_WEIGHT,
                                dist_type);
    }
    // ---（qi, wi) ---
    else if (parameters.get<std::string>("exc_type") == "recall_search") {   // recall_search (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_HNSW_FUSION, &graph_file[0]);
        
        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_NONE_FUSION, xmt::TYPE::ROUTER_HNSW_FUSION, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_ALL_INDEX, xmt::TYPE::LOAD_WEIGHT,
                                dist_type); 
    }
    // ---（qi, W) ---
    else if (parameters.get<std::string>("exc_type") == "all_recall_search") {   // all_recall_search (various fixed L)
        smart_builder -> load(parameters);
    
        smart_builder -> load_graph(xmt::TYPE::INDEX_HNSW_FUSION, &graph_file[0]);
        
        smart_builder -> load_search_weight();

        smart_builder -> search(xmt::TYPE::SEARCH_ENTRY_NONE_FUSION, xmt::TYPE::ROUTER_HNSW_FUSION, 
                                xmt::TYPE::L_RECALL_SEARCH_CONTROL_ALL_INDEX, xmt::TYPE::LOADED_ALL_WEIGHT,
                                dist_type); 
    }
    else {
        std::cout << "exc_type input error10!" << std::endl;
    }
};
# endif // INDEXCODEFROMMAIN_H