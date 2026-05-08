#ifndef XMT_BUILDER_H
#define XMT_BUILDER_H

#include "index.h"


namespace xmt {
    class MultiIndexBuilder {
        public:
            explicit MultiIndexBuilder(const unsigned num_threads) {
                smart_final_index_ = new MultiIndex();
                omp_set_num_threads(num_threads);
                std::cout << "\n ======================================" << std::endl;
                std::cout << " = omp_set_num_threads(" << num_threads << ")" << std::endl;
                std::cout << " ======================================\n" << std::endl;
            }
    
            virtual ~MultiIndexBuilder() {
                delete smart_final_index_;
            }
    
            MultiIndexBuilder *load(Parameters &parameters);

            MultiIndexBuilder *preliminary(Parameters &parameters, TYPE group_type, TYPE param_type);

            MultiIndexBuilder *init(TYPE type, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

            MultiIndexBuilder *refine(TYPE type, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

            MultiIndexBuilder *connectivity_enforcer(TYPE type, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    
            MultiIndexBuilder *save_graph(TYPE type, char *graph_file);
    
            MultiIndexBuilder *load_graph(TYPE type, char *graph_file);

            MultiIndexBuilder *load_search_weight();
    
            MultiIndexBuilder *search(TYPE entry_type, TYPE route_type, TYPE L_type, TYPE weight_type, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    
            MultiIndexBuilder *get_ground_truth(TYPE weight_type, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    
            MultiIndex *getFinalIndex() {
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
            MultiIndex *smart_final_index_;
    
            std::chrono::high_resolution_clock::time_point s;
            std::chrono::high_resolution_clock::time_point e;
    };
}

#endif //XMT_BUILDER_H
