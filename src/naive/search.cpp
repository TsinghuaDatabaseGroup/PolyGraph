# include <vector>
# include <iostream>
#include <fstream>

# include <naive/search.h>


/**
 *      SearchFor_v1_weight():
 * only for one multi-vector query, with given weight. 
 */
void SearchFor_v1_weight(std::vector<float*> &base_data_list, std::vector<float*> &query_data_list, std::vector<float> &weight,
                std::vector<unsigned int> &result, std::vector<float> &score_result, 
                unsigned int num, std::vector<unsigned int> dim_list, unsigned int query, xmt::TYPE dist_type,
                unsigned K) {
    result.clear();
    score_result.clear();

    xmt::CandidateHeap2 cands;
    if (dist_type == xmt::TYPE::DIST_EUCLIDEAN) {
        for (unsigned int j = 0; j < num; j++) {
            float total_score = 0;
            
            for (unsigned col = 0; col < query_data_list.size(); col++) {
                total_score += weight[col] * compare(query_data_list[col]+ query * dim_list[col] * sizeof(float)/4,
                                    base_data_list[col] + j * dim_list[col] * sizeof(float)/4,
                                    dim_list[col] * sizeof(float)/4);
            }
            xmt::Candidate2<float> c(j, total_score);
            cands.insert(c);
            if (cands.size() > K)cands.erase(cands.begin());
        }
    } else if (dist_type == xmt::TYPE::DIST_COS) {
        for (unsigned int j = 0; j < num; j++) {
            float total_score = 0;
            for (int col = 0; col < (int)query_data_list.size(); col++) {
                total_score += weight[col] * compare_cosDist(query_data_list[col]+ query * dim_list[col] * sizeof(float)/4,
                                    base_data_list[col] + j * dim_list[col] * sizeof(float)/4,
                                    dim_list[col] * sizeof(float)/4);
            }
            xmt::Candidate2<float> c(j, total_score);
            cands.insert(c);
            if (cands.size() > K)cands.erase(cands.begin());
        }
    } else {
        for (unsigned int j = 0; j < num; j++) {
            float total_score = 0;
            for (int col = 0; col < (int)query_data_list.size(); col++) {
                total_score += weight[col] * compare_cosSim(query_data_list[col]+ query * dim_list[col] * sizeof(float)/4,
                                    base_data_list[col] + j * dim_list[col] * sizeof(float)/4,
                                    dim_list[col] * sizeof(float)/4);
            }
            xmt::Candidate2<float> c(j, total_score);
            cands.insert(c);
            if (cands.size() > K)cands.erase(cands.begin());
        }
    }
    
    auto it = cands.rbegin();
    for(unsigned int j = 0; it != cands.rend() && j < K; it++, j++){
        result.push_back(it->row_id);
        score_result.push_back(it->distance);
    }
};

// for bench multi-vector query, with given weight(每个query可以对应不一样的weight)；给MultiIndex使用 
void ground_truth_diffWeight(std::vector<float*> base_data_list, std::vector<float*> query_data_list, std::vector<std::vector<float>> weight_list,
                unsigned int num, unsigned int num_query, std::vector<unsigned int> dim_list, 
                std::vector<std::vector<unsigned>> &res, std::vector<std::vector<float>> &dist_res,
                xmt::TYPE dist_type, unsigned K, bool hint) {
    res.clear();
    res.resize(num_query);
    dist_res.clear();
    dist_res.resize(num_query);

    std::cout << "___ Getting Ground Truth by 'ground_truth_diffWeight()'___" << std::endl;
    
    // do search for each query
    auto s1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for  
    for (unsigned i = 0; i < num_query; i++) {
        SearchFor_v1_weight(base_data_list, query_data_list, weight_list[i], res[i], dist_res[i], num, dim_list, i, dist_type, K);
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e1 - s1;
    if (hint) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Pure total naive query time(search): " << diff.count() << "\n";
        std::cout << "-- k-max NN result:" << std::endl;
        std::cout << "-- Ground-truths(naive search)" << std::endl;
        for (unsigned i = 0; i < 2; i++) {
            std::cout << "- For query " << i << "\n     ";
            for (unsigned j = 0; j < K; j++) {
                std::cout << res[i][j] << ", ";
            }
            std::cout << std::endl;
            std::cout << "dist: ";
            for (unsigned j = 0; j < K; j++) {
                std::cout << dist_res[i][j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------------------------------------\n\n" << std::endl;
    }
};