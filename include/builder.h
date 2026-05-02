//
// Created by mengtong-x on 2024/06/23.
//

#ifndef WEAVESS_BUILDER_H
#define WEAVESS_BUILDER_H

#include "index.h"


namespace weavess {
    // class IndexBuilder {
    // public:
    //     explicit IndexBuilder(const unsigned num_threads) {
    //         final_index_ = new Index();
    //         omp_set_num_threads(num_threads);
    //         std::cout << "\n ======================================" << std::endl;
    //         std::cout << " = omp_set_num_threads(" << num_threads << ")" << std::endl;
    //         std::cout << " ======================================\n" << std::endl;
    //     }

    //     virtual ~IndexBuilder() {
    //         delete final_index_;
    //     }

    //     IndexBuilder *load(char *data_file, char *query_file, char *ground_file, Parameters &parameters);

    //     IndexBuilder *load_graph(TYPE type, char *graph_file);

    //     IndexBuilder *search_weight(TYPE entry_type, TYPE route_type, TYPE L_type, std::vector<std::vector<float>> &recall_result,
    //                                 std::vector<std::vector<float>> &latency_result, std::string neigh_save_name = "NULL", TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);

    //     Index *getFinalIndex() {
    //         return final_index_;
    //     }

    //     unsigned int getBaseLen() const {
    //         return final_index_->getBaseLen();
    //     }

    //     unsigned int getQueryLen() const {
    //         return final_index_->getQueryLen();
    //     }

    //     unsigned int getBaseDim() const {
    //         return final_index_->getBaseDim();
    //     }
        
    // private:
    //     Index *final_index_;

    //     std::chrono::high_resolution_clock::time_point s;
    //     std::chrono::high_resolution_clock::time_point e;
    // };




    // ---------------------------------------
    // SmartIndex: SmartIndexBuilder (2025.04.17)
    //----------------------------------------
    class SmartIndexBuilder {
        public:
            explicit SmartIndexBuilder(const unsigned num_threads) {
                smart_final_index_ = new SmartIndex();
                omp_set_num_threads(num_threads);
                std::cout << "\n ======================================" << std::endl;
                std::cout << " = omp_set_num_threads(" << num_threads << ")" << std::endl;
                std::cout << " ======================================\n" << std::endl;
            }
    
            virtual ~SmartIndexBuilder() {
                delete smart_final_index_;
            }
    
            SmartIndexBuilder *load(Parameters &parameters);

            SmartIndexBuilder *preliminary(Parameters &parameters, TYPE group_type, TYPE param_type);

            SmartIndexBuilder *init(TYPE type, TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);

            SmartIndexBuilder *refine(TYPE type, TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);

            SmartIndexBuilder *connectivity_enforcer(TYPE type, TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);
    
            SmartIndexBuilder *save_graph(TYPE type, char *graph_file);
    
            SmartIndexBuilder *load_graph(TYPE type, char *graph_file);

            SmartIndexBuilder *load_search_weight();
    
            SmartIndexBuilder *search(TYPE entry_type, TYPE route_type, TYPE L_type, TYPE weight_type, TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);
    
            SmartIndexBuilder *search_weight(TYPE entry_type, TYPE route_type, TYPE L_type, std::vector<std::vector<float>> &recall_result,
                                        std::vector<std::vector<float>> &latency_result, std::string neigh_save_name = "NULL", TYPE dist_type = weavess::TYPE::DIST_EUCLIDEAN);
    
            SmartIndex *getFinalIndex() {
                return smart_final_index_;
            }

            std::chrono::duration<double> GetBuildTime() { return e - s; }

            Parameters &getParam() {
                return smart_final_index_->getParam();
            };
    
            unsigned int getBaseLen() const {
                return smart_final_index_->getBaseLen();
            }
    
            unsigned int getQueryLen() const {
                return smart_final_index_->getQueryLen();
            }
    
            unsigned int getGroundLen() const {
                return smart_final_index_->getGroundLen();
            }
    
            unsigned int getBaseDim(int field) const {
                return smart_final_index_->getBaseDim(field);
            }
    
            unsigned int getQueryDim(int field) const {
                return smart_final_index_->getQueryDim(field);
            }
    
            unsigned int getGroundDim() const {
                return smart_final_index_->getGroundDim();
            }
    
            void resetDistCount() {
                smart_final_index_->resetDistCount();
            }
    
            void resetHopCount() {
                smart_final_index_->resetHopCount();
            }

            void showPresentTime() {
                auto now = std::chrono::system_clock::now();
                std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
                std::tm* now_tm = std::localtime(&now_time_t);
                std::cout << "-- Record time is: " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
            }
        private:
            SmartIndex *smart_final_index_;
    
            std::chrono::high_resolution_clock::time_point s;
            std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //WEAVESS_BUILDER_H
