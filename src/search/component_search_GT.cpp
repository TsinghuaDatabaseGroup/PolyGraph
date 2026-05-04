#include "component.h"
#include "naive/search.h"


// ======================================
// get Ground Truth
// ======================================

namespace xmt {
    void ComponentGroundTruth_multi::GroundInner_multi(unsigned K, TYPE dist_type)
    {
        // ## 1. search to get ground-truths
        std::vector<std::vector<unsigned>> res_list;
        std::vector<std::vector<float>> dist_res_list;
        ground_truth_diffWeight(smart_index->getBaseDataList(), smart_index->getQueryDataList(), smart_index->getSearchWeight(),
                                smart_index->getBaseLen(), smart_index->getQueryLen(), smart_index->getBaseDimList(),
                                res_list, dist_res_list, dist_type, K);

        // ## 2. 展示ground-truth
        std::cout << "--- For Check ---" << std::endl;
        std::cout << "-- Ground-truths" << std::endl;
        for (unsigned i = 0; i < 2; i++)
        {
            std::cout << "- For query " << i << "\n     ";
            for (unsigned j = 0; j < K; j++)
            {
                std::cout << res_list[i][j] << ", ";
            }
            std::cout << std::endl;
            std::cout << "dist: ";
            for (unsigned j = 0; j < K; j++)
            {
                std::cout << dist_res_list[i][j] << ", ";
            }
            std::cout << std::endl;
        }

        // ## 3. save ground-truth
        unsigned *ground = new unsigned[K * smart_index->getQueryLen()];
        unsigned pos = 0;
        for (unsigned i = 0; i < smart_index->getQueryLen(); i++)
        {
            for (unsigned j = 0; j < K; j++)
            {
                ground[pos++] = res_list[i][j];
            }
        }
        smart_index->reSetGroundData(ground, K);
    }

    void ComponentGroundTruth_multi::GroundInner_multi_load(unsigned w, unsigned K, TYPE dist_type)
    {
        // ## 1. Read-Ground-truth from fiel to get ground-truths
        // ground_data
        std::string ground_path = "../dataset/Ground-truth/" + smart_index->getParam().get<std::string>("dataset") + "/" + std::to_string(w) + "-output.ivecs";
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        std::cout << "### Ground-truth from: " << ground_path << std::endl;
        xmt::load_data<unsigned>(ground_path.c_str(), ground_data, ground_num, ground_dim);

        if (ground_num != smart_index->getQueryLen() || ground_dim < K)
        {
            GroundInner_multi(K, dist_type);
        }
        else
        {
            smart_index->setGroundData(ground_data);
            smart_index->setGroundLen(ground_num);
            smart_index->setGroundDim(ground_dim);
        }

        // ## 2. 展示ground-truth
        std::cout << "--- For Check ---" << std::endl;
        std::cout << "-- Ground-truths" << std::endl;
        for (unsigned i = 0; i < 2; i++) {
            std::cout << "- For query " << i << "\n     ";
            for (unsigned j = 0; j < K; j++) {
                std::cout << ground_data[i*K + j] << ", ";
            }
            std::cout << std::endl;
        }
    }
}
