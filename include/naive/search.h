#ifndef XMT_SEARCH_H
#define XMT_SEARCH_H

#include <vector>
#include <iostream>
#include <fstream>

#include <builder.h>

void SearchFor_v1_weight(std::vector<float*> &base_data_list, std::vector<float*> &query_data_list, std::vector<float> &weight,
                std::vector<unsigned int> &result, std::vector<float> &score_result, 
                unsigned int num, std::vector<unsigned int> dim_list, unsigned int query, xmt::TYPE dist_type,
                unsigned K = 20);
void ground_truth_diffWeight(std::vector<float*> base_data_list, std::vector<float*> query_data_list, std::vector<std::vector<float>> &weight_list,
                unsigned int num, unsigned int num_query, std::vector<unsigned int> dim_list, 
                std::vector<std::vector<unsigned>> &res, std::vector<std::vector<float>> &dist_res,
                xmt::TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN, unsigned K = 20, bool hint = true);

#endif //XMT_SEARCH_H