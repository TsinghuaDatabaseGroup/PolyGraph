#ifndef XMT_COMPONENT_H
#define XMT_COMPONENT_H

#include "index.h"
#include "builder.h"
#include "policy.h"
#include <map>


#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace xmt {
    template<typename T>
    void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim) {
        std::ifstream in(filename, std::ios::binary);
        // ### also try to load from --TXT-- of a certained format
        if (!in.is_open()) {
            std::cout << " I am at pos load_data.1" << std::endl;
            std::ifstream fileStream;
            fileStream.open(filename, std::ios::in);
 
            if (fileStream.fail())
            {
                std::cout << " I am still here... read file fail of filename: " << filename << std::endl;
                throw std::logic_error("read file fail");
                std::cerr << "open file error-11" << std::endl;
                exit(-1);
            } else
            {
                int row = 0;
 
                std::string tmp;
                unsigned count = 0;// 行数计数器

                // 第一行存了num，dim
                getline(fileStream, tmp, '\n');
                std::string str_tmp;
                std::istringstream is(tmp);
                is>>str_tmp;
                num = static_cast<unsigned>(std::stof(str_tmp));
                is>>str_tmp;
                dim = static_cast<unsigned>(std::stof(str_tmp));
                data = new T[num * dim];
                while (getline(fileStream, tmp, '\n'))//读取一行
                {
                    std::vector<float> tmpV{};
                    std::istringstream is(tmp);
                    for(unsigned i = 0; i < dim; i++){
                        std::string str_tmp;
                        is>>str_tmp;
                        data[count * dim + i] = static_cast<unsigned>(std::stof(str_tmp));
                    }
                    count++;
                }
                std::cout << num << ", " << dim << "; " << count << ", " << std::endl;
                std::cout << "Successfully read " << filename << std::endl;
                fileStream.close();
            }
        }
        // ### 读ivecs，或者fvecs
        else {
            in.read((char *) &dim, 4);
            in.seekg(0, std::ios::end);
            std::ios::pos_type ss = in.tellg();
            auto f_size = (size_t) ss;
            num = (unsigned) (f_size / (dim + 1) / 4);
            data = new T[num * dim];
            in.seekg(0, std::ios::beg);
            for (size_t i = 0; i < num; i++) {
                in.seekg(4, std::ios::cur);
                in.read((char *) (data + i * dim), dim * sizeof(T));
            }
            in.close();
        }
    }

    inline void write_fvecs(const std::string& path, const float* data, std::size_t num, std::size_t dim)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("open failed: " + path);

        const int32_t d32 = static_cast<int32_t>(dim);
        const std::size_t stride_bytes = dim * sizeof(float);

        for (std::size_t i = 0; i < num; ++i) {
            out.write(reinterpret_cast<const char*>(&d32), sizeof(d32));     // 写维度头
            out.write(reinterpret_cast<const char*>(data + i * dim),         // 写向量
                    static_cast<std::streamsize>(stride_bytes));
            if (!out) throw std::runtime_error("write failed at vector " + std::to_string(i));
        }
    }















    // ---------------------------------------------------------------------
    // For MultiIndex
    // ---------------------------------------------------------------------
    class Component_smart {
    public:
        explicit Component_smart(MultiIndex *smart_index) : smart_index(smart_index) {}

        virtual ~Component_smart() { delete smart_index; }

    protected:
        MultiIndex *smart_index = nullptr;
    };

    // ===== load data ===== 
    //                  ---> component_load.cpp
    class ComponentLoad_smart : public Component_smart {
    public:
        explicit ComponentLoad_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void LoadInner_smart(Parameters &parameters);
        
        void set_path_from_txt(std::string txt_path, std::vector<std::string> &base_path_list, std::vector<std::string> &query_path_list, bool show_summary = true);
    };



    // ******************
    // Construction
    // *******************
    // ===== initialization -- preliminary ===== 
    //                  ---> component_init.cpp
    class ComponentPreliminary_smart : public Component_smart {
    public:
        explicit ComponentPreliminary_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        // seperate group
        void SeperateInner_smart_ClusterGroup(Parameters &parameters);

        // baseline: 
        void SeperateInner_smart_GroupEquNoTotal(Parameters &parameters);
        void SeperateInner_smart_GroupFusion(Parameters &parameters);
        void SeperateInner_smart_GroupAllWeight(Parameters &parameters);

        // assign L_refine, R_refine
        void PrepareParameter_smart_Equal(Parameters &parameters);
    private:
        float calMaxNorm(unsigned len, std::vector<float*> base_data_list, std::vector<unsigned> dim_list,
                            std::vector<int> &relaField, xmt::TYPE dist_type);

        float cal_corr(const std::vector<float>& X, const std::vector<float>& Y) {
            if (X.size() != Y.size() || X.empty()) {
                throw std::invalid_argument("Input vectors must have the same non-zero size.");
            }
        
            size_t n = X.size();
            float mean_X = std::accumulate(X.begin(), X.end(), 0.0f) / n;
            float mean_Y = std::accumulate(Y.begin(), Y.end(), 0.0f) / n;
        
            float numerator = 0.0f;
            float denom_X = 0.0f;
            float denom_Y = 0.0f;
        
            for (size_t i = 0; i < n; ++i) {
                float dx = X[i] - mean_X;
                float dy = Y[i] - mean_Y;
                numerator += dx * dy;
                denom_X += dx * dx;
                denom_Y += dy * dy;
            }
        
            if (denom_X == 0.0f || denom_Y == 0.0f) {
                // 避免除以0，说明某个向量标准差为0
                return 0.0f;
            }
        
            return numerator / (std::sqrt(denom_X) * std::sqrt(denom_Y));
        };
    };





    // ===== initialization -- initial graph =====
    //                  ---> construct/component_init.cpp
    class ComponentInit_smart : public Component_smart {
    public:
        explicit ComponentInit_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void InitInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;
    };
    
    class ComponentInitRand_smart : public ComponentInit_smart {
    public:
        explicit ComponentInitRand_smart(MultiIndex *smart_index) : ComponentInit_smart(smart_index) {}

        void InitInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

    private:
        void InitInner_smart_4group(int group, std::mt19937 &rng, std::vector<std::vector<unsigned>> &initON, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void SetConfigs_smart(int group);
    };
    
    class ComponentInitHNSW_Fusion : public ComponentInit_smart {
    public:
        explicit ComponentInitHNSW_Fusion(MultiIndex *smart_index) : ComponentInit_smart(smart_index) {}

        void InitInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

    private:
        void SetConfigs();

        void Build(bool reverse, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        static int GetRandomSeedPerThread();

        int GetRandomNodeLevel();

        void InsertNode(MultiIndex::HnswNode *qnode, MultiIndex::VisitedList *visited_list, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void SearchAtLayer(MultiIndex::HnswNode *qnode, MultiIndex::HnswNode *enterpoint, int level,
                           MultiIndex::VisitedList *visited_list, std::priority_queue<MultiIndex::FurtherFirst> &result, 
                           TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void Link(MultiIndex::HnswNode *source, MultiIndex::HnswNode *target, int level, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };





    // ===== initialization -- entry-point selection ===== 
    //                  ---> construct/component_init.cpp
    class ComponentRefineEntry_smart : public Component_smart {
    public:
        explicit ComponentRefineEntry_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void EntryInner_smart() = 0;

        virtual ~ComponentRefineEntry_smart() { 
            for (auto ptr : center_) {
                delete[] ptr;
            } 
    }

    protected:
        std::vector<float *> center_;
    };

    class ComponentRefineEntryCentroid_smart : public ComponentRefineEntry_smart {
    public:
        explicit ComponentRefineEntryCentroid_smart(MultiIndex *smart_index) : ComponentRefineEntry_smart(smart_index) {}

        void EntryInner_smart() override;

        void EntryInner_smart_4group(int group);

    private:
        void get_exact_neighbor_smart_4group(int group, MultiIndex::Neighbor &nn, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };




    // ===== refine graph: Neighborhood Construction ===== 
    //                  ---> construct/component_refine.cpp
    class ComponentRefine_smart : public Component_smart {
    public:
        explicit ComponentRefine_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void RefineInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;
    };
        
    class ComponentRefineSmart : public ComponentRefine_smart {
    public:
        explicit ComponentRefineSmart(MultiIndex *smart_index) : ComponentRefine_smart(smart_index) {}

        void RefineInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

    private:
        void SetConfigs_smart(int group, float alpha);

        void RefineInner_smart_4group(int group, float alpha, bool hint, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void Link_smart_4group(int group, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void InterInsert_smart_4group_insert(unsigned n, int group, std::vector<std::mutex> &locks,
                                             std::vector<int> &orignNum,TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void InterInsert_smart_4group_prune(unsigned n, int group, std::vector<std::mutex> &locks,
                                      TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };

    class ComponentRefineSmart_Oracle : public ComponentRefine_smart {
    public:
        explicit ComponentRefineSmart_Oracle(MultiIndex *smart_index) : ComponentRefine_smart(smart_index) {}

        void RefineInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

    private:
        void SetConfigs_smart(int group, float alpha);

        void RefineInner_smart_4group(int group, float alpha, bool hint, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void Link_smart_4group(int group, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void InterInsert_smart_4group_insert(unsigned n, int group, std::vector<std::mutex> &locks,
                                             std::vector<int> &orignNum,TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);

        void InterInsert_smart_4group_prune(unsigned n, int group, std::vector<std::mutex> &locks,
                                      TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };

    


    // ===== Candidate Neighbor Acquisition ===== 
    //                  ---> construct/component_candidate.cpp
    class ComponentCandidate_smart : public Component_smart {
    public:
        explicit ComponentCandidate_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void CandidateInner_smart_4group_STAR(unsigned query, unsigned enter, int group, boost::dynamic_bitset<> flags,
                                    std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0; 
        virtual void CandidateInner_smart_4group_STAR_allIndex(unsigned query, unsigned enter, int group, boost::dynamic_bitset<> flags,
                            std::vector<MultiIndex::SimpleNeighbor> &result, std::vector<std::mutex> &locks, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;                                    
    };

    class ComponentCandidateAGS_smart : public ComponentCandidate_smart {
    public:
        explicit ComponentCandidateAGS_smart(MultiIndex *smart_index) : ComponentCandidate_smart(smart_index) {}

        void CandidateInner_smart_4group_STAR(unsigned query, unsigned enter, int group, boost::dynamic_bitset<> flags,
                            std::vector<MultiIndex::SimpleNeighbor> &result, std::vector<std::mutex> &locks, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
        void CandidateInner_smart_4group_STAR_allIndex(unsigned query, unsigned enter, int group, boost::dynamic_bitset<> flags,
                            std::vector<MultiIndex::SimpleNeighbor> &result, std::vector<std::mutex> &locks, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
    };





    // ===== graph prune ===== 
    //                  ---> construct/component_prune.cpp
    class ComponentPrune_smart : public Component_smart {
    public:
        explicit ComponentPrune_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void PruneInner_smart_4group_withOthers(unsigned query, int group,  
                                std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks,
                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;

        virtual void PruneInner_Fusion(unsigned q, unsigned range,
                        std::vector<MultiIndex::SimpleNeighbor> &pool, MultiIndex::SimpleNeighbor *cut_graph_, 
                        TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;

        void Hnsw2Neighbor_Fusion(unsigned query, unsigned range, std::priority_queue<MultiIndex::FurtherFirst> &result,
                            TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) {
            int n = result.size();
            std::vector<MultiIndex::SimpleNeighbor> pool(n);
            std::unordered_map<int, MultiIndex::HnswNode *> tmp;

            for (int i = n - 1; i >= 0; i--) {
                MultiIndex::FurtherFirst f = result.top();
                pool[i] = MultiIndex::SimpleNeighbor(f.GetNode()->GetId(), f.GetDistance());
                tmp[f.GetNode()->GetId()] = f.GetNode();
                result.pop();
            }

            boost::dynamic_bitset<> flags;

            auto *cut_graph_ = new MultiIndex::SimpleNeighbor[smart_index->getBaseLen() * range];

            PruneInner_Fusion(query, range, pool, cut_graph_, dist_type);

            for (unsigned j = 0; j < range; j++) {
                if (cut_graph_[range * query + j].distance == -1) break;

                result.push(MultiIndex::FurtherFirst(tmp[cut_graph_[range * query + j].id], cut_graph_[range * query + j].distance));
            }

            delete[] cut_graph_;

            std::vector<MultiIndex::SimpleNeighbor>().swap(pool);
            std::unordered_map<int, MultiIndex::HnswNode *>().swap(tmp);
        }
    };

    class ComponentPrunePolyGraph_smart : public ComponentPrune_smart {
    public:
        explicit ComponentPrunePolyGraph_smart(MultiIndex *smart_index) : ComponentPrune_smart(smart_index) {}

        void PruneInner_smart_4group_withOthers(unsigned query, int group, 
                        std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks,
                        TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

        // 占位符
        // void PruneInner_Fusion(unsigned q, unsigned range,
        //                 std::vector<MultiIndex::SimpleNeighbor> &pool, MultiIndex::SimpleNeighbor *cut_graph_, 
        //                 TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
        void PruneInner_Fusion(unsigned q, unsigned range,
                        std::vector<MultiIndex::SimpleNeighbor> &pool, MultiIndex::SimpleNeighbor *cut_graph_, 
                        TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) {};
        // 占位符end
    };

    class ComponentPruneHeuristic_smart : public ComponentPrune_smart {
    public:
        explicit ComponentPruneHeuristic_smart(MultiIndex *smart_index) : ComponentPrune_smart(smart_index) {}

        void PruneInner_Fusion(unsigned q, unsigned range,
                        std::vector<MultiIndex::SimpleNeighbor> &pool, MultiIndex::SimpleNeighbor *cut_graph_, 
                        TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

        // 占位符
        // void PruneInner_smart_4group_withOthers(unsigned query, int group, 
        //                 std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks,
        //                 TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
        void PruneInner_smart_4group_withOthers(unsigned query, int group, 
                        std::vector<MultiIndex::SimpleNeighbor> &pool, std::vector<std::mutex> &locks,
                        TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) {};
        // 占位符end
    };




    







    // ===== graph connectivity enforcer ===== 
    //                  ---> construct/component_CE.cpp
    class ComponentConnectEnforcer_smart : public Component_smart {
    public:
        explicit ComponentConnectEnforcer_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void ConnectEnforcerInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;
    };

    class ComponentRelaConnectEnforcer : public ComponentConnectEnforcer_smart {
    public:
        explicit ComponentRelaConnectEnforcer(MultiIndex *smart_index) : ComponentConnectEnforcer_smart(smart_index) {}

        void ConnectEnforcerInner_smart(TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;

    private:
        bool FindIsolate(unsigned ep, std::vector<int> relaGroup, std::set<unsigned> &isolate_ids);

        void ConnectAndSpread(int group, const std::vector<int> relaGroup, std::set<unsigned> &isolate_ids);

        MultiIndex::SimpleNeighbor get_connect_neighbor_limited_randEP(unsigned query, int group, std::vector<int> relaGroup, unsigned &numNoPos, unsigned &numUnconnected,
                                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);                                                                                                            
    };

    







    // ******************
    // Search
    // *******************
    // ===== search entry ===== 
    //                  ---> search/component_search_entry.cpp
    class ComponentSearchEntry_smart : public Component_smart {
    public:
        explicit ComponentSearchEntry_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void SearchEntryInner_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, 
                                            std::vector<int> &needCalField, std::vector<int> &needIndeces, 
                                            std::vector<unsigned> &checkEps, boost::dynamic_bitset<> &flags,
                                            TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;
    };

    class ComponentSearchEntryNone_Fusion : public ComponentSearchEntry_smart {
    public:
        explicit ComponentSearchEntryNone_Fusion(MultiIndex *smart_index) : ComponentSearchEntry_smart(smart_index) {}

        void SearchEntryInner_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, 
                                    std::vector<int> &needCalField, std::vector<int> &needIndeces, 
                                    std::vector<unsigned> &checkEps, boost::dynamic_bitset<> &flags, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
    };

    class ComponentSearchEntryCentroid_smart : public ComponentSearchEntry_smart {
    public:
        explicit ComponentSearchEntryCentroid_smart(MultiIndex *smart_index) : ComponentSearchEntry_smart(smart_index) {}

        void SearchEntryInner_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, 
                                    std::vector<int> &needCalField, std::vector<int> &needIndeces, 
                                    std::vector<unsigned> &checkEps, boost::dynamic_bitset<> &flags, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
    };



    



    // ===== search route ===== 
    //                  ---> search/component_search_route.cpp
    class ComponentSearchRoute_smart : public Component_smart {
    public:
        explicit ComponentSearchRoute_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        virtual void RouteInner_dist_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, std::vector<unsigned> &res, std::vector<float> &dist_res, 
                                            std::vector<int> &needCalField, std::vector<int> &needIndeces, boost::dynamic_bitset<> &flags, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) = 0;
    };

    class ComponentSearchRouteGreedy_smart : public ComponentSearchRoute_smart {
    public:
        explicit ComponentSearchRouteGreedy_smart(MultiIndex *smart_index) : ComponentSearchRoute_smart(smart_index) {}

        void RouteInner_dist_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, std::vector<unsigned> &res, std::vector<float> &dist_res, 
                                    std::vector<int> &needCalField, std::vector<int> &needIndeces, boost::dynamic_bitset<> &flags, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
    };

    class ComponentSearchRouteHNSW_Fusion : public ComponentSearchRoute_smart {
    public:
        explicit ComponentSearchRouteHNSW_Fusion(MultiIndex *smart_index) : ComponentSearchRoute_smart(smart_index) {}

        void RouteInner_dist_smart(unsigned query, std::vector<MultiIndex::Neighbor> &pool, std::vector<unsigned> &res, std::vector<float> &dist_res, 
                                    std::vector<int> &needCalField, std::vector<int> &needIndeces, boost::dynamic_bitset<> &flags, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN) override;
    private:
        void SearchAtLayer(unsigned qnode, MultiIndex::HnswNode *enterpoint, int level,
                           std::vector<float> &weight, std::vector<int> &needCalField, MultiIndex::VisitedList *visited_list,
                           std::priority_queue<MultiIndex::FurtherFirst> &result, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };
    
    
    
    
    
    
    
    
    // ===== search flow ===== 
    //                  ---> search/component_search_flow.cpp
    class ComponentSearchFlow_smart : public Component_smart {
    public:
        explicit ComponentSearchFlow_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}
    };

    class ComponentSearchFlowLoadWeight_smart : public ComponentSearchFlow_smart {
    public:
        explicit ComponentSearchFlowLoadWeight_smart(MultiIndex *smart_index) : ComponentSearchFlow_smart(smart_index) {}
        
        void FlowInner_WeightOnce_smart_FallbackIntersect(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                    float &recall, float&latency, float &hop, float &distCount, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnce_smart_intersect(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                    float &recall, float&latency, float &hop, float &distCount, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnce_smart_exactRepre(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                    float &recall, float&latency, float &hop, float &distCount, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnce_smart_allIndex(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                    float &recall, float&latency, float &hop, float &distCount, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnce_smart_FallbackIntersect_forDiff_wi(unsigned K, unsigned L, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                    float &recall, float&latency, float &hop, float &distCount, 
                                    TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        


        void FlowInner_WeightOnceControL_smart_FallbackIntersect(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnceControL_smart_intersect(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnceControL_smart_exactRepre(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnceControL_smart_allIndex(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void FlowInner_WeightOnceControL_smart_FallbackIntersect_forDiff_wi(unsigned K, std::vector<unsigned> &LRate, ComponentSearchEntry_smart *a, ComponentSearchRoute_smart *b, 
                                                std::vector<float> &recall_list, std::vector<float> &latency_list,
                                                std::vector<float> &hop_list, std::vector<float> &distCount_list,
                                                TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);


    private:
        void prepareForSearch_smart_SetNeedCalField(unsigned query,std::vector<int> &needCalField, std::vector<int> &needIndeces, std::vector<unsigned> &checkEps);
        void prepareForSearch_smart_SavedAGS_FallbackIntersect(unsigned query,std::vector<int> &needCalField, std::vector<int> &needIndeces, std::vector<unsigned> &checkEps);
        void prepareForSearch_smart_intersect(unsigned query,std::vector<int> &needCalField, std::vector<int> &needIndeces, std::vector<unsigned> &checkEps);
        void prepareForSearch_smart_exactRepre(unsigned query,std::vector<int> &needCalField, std::vector<int> &needIndeces, std::vector<unsigned> &checkEps);
        void prepareForSearch_smart_allIndex(unsigned query,std::vector<int> &needCalField, std::vector<int> &needIndeces, std::vector<unsigned> &checkEps);
    };








    // ===== ground-truth ===== 
    //                  ---> search/component_GT.cpp
    class ComponentGroundTruth_smart : public Component_smart {
    public:
        explicit ComponentGroundTruth_smart(MultiIndex *smart_index) : Component_smart(smart_index) {}

        void GroundInner_smart(unsigned K, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
        void GroundInner_smart_load(unsigned w, unsigned K, TYPE dist_type = xmt::TYPE::DIST_EUCLIDEAN);
    };
}
#endif //XMT_COMPONENT_H
