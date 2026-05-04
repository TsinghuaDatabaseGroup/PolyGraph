#ifndef XMT_INDEX_H
#define XMT_INDEX_H

#define PARALLEL

#include <atomic>
#include <omp.h>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>
#include <set>
#include<map>
#include <chrono>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <mm_malloc.h>
#include <stdlib.h>


#include "util.h"
#include "policy.h"
#include "distance.h"
#include "parameters.h"


namespace xmt {

    class NNDescent {
    public:
        unsigned K;
        unsigned S;
        unsigned R;
        unsigned L;
        unsigned ITER;

        struct Neighbor {
            unsigned id;
            float distance;
            bool flag;
            int group = -1;

            Neighbor() = default;

            Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}
            Neighbor(unsigned id, float distance, int group, bool f) : id{id}, distance{distance}, flag(f), group(group) {}

            inline bool operator<(const Neighbor &other) const {
                return distance < other.distance;
            }

            inline bool operator>(const Neighbor &other) const {
                return distance > other.distance;
            }
        };

        typedef std::lock_guard<std::mutex> LockGuard;

        struct nhood {
            std::mutex lock;
            std::vector<Neighbor> pool;
            unsigned M;

            std::vector<unsigned> nn_old;
            std::vector<unsigned> nn_new;
            std::vector<unsigned> rnn_old;
            std::vector<unsigned> rnn_new;

            nhood() {}

            nhood(unsigned l, unsigned s) {
                M = s;
                nn_new.resize(s * 2);
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
                M = s;
                nn_new.resize(s * 2);
                GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
                nn_new.reserve(s * 2);
                pool.reserve(l);
            }

            nhood(const nhood &other) {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            void insert(unsigned id, float dist) {
                LockGuard guard(lock);
                if (dist > pool.front().distance) return;
                for (unsigned i = 0; i < pool.size(); i++) {
                    if (id == pool[i].id)return;
                }
                if (pool.size() < pool.capacity()) {
                    pool.push_back(Neighbor(id, dist, true));
                    std::push_heap(pool.begin(), pool.end());
                } else {
                    std::pop_heap(pool.begin(), pool.end());
                    pool[pool.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(pool.begin(), pool.end());
                }

            }

            template<typename C>
            void join(C callback) const {
                for (unsigned const i: nn_new) {
                    for (unsigned const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
            }
        };

        static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
            // find the location to insert
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance) {
                memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance) {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1) {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)right = mid;
                else left = mid;
            }

            while (left > 0) {
                if (addr[left].distance < nn.distance) break;
                if (addr[left].id == nn.id) return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
            memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        static inline int InsertIntoPool(Neighbor *addr, unsigned size, unsigned K, Neighbor nn) {
            // find the location to insert
            if (size < K) {
                new (&addr[size]) Neighbor(nn);  // 在 addr[size] 的位置构造 nn
                return size;
            }
            int left = 0, right = K - 1;
            if (addr[left].distance > nn.distance) {
                memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
                addr[left] = nn;
                return left;
            }
            if (addr[right].distance < nn.distance) {
                addr[K] = nn;
                return K;
            }
            while (left < right - 1) {
                int mid = (left + right) / 2;
                if (addr[mid].distance > nn.distance)right = mid;
                else left = mid;
            }

            while (left > 0) {
                if (addr[left].distance < nn.distance) break;
                if (addr[left].id == nn.id) return K + 1;
                left--;
            }
            if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
            memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
            addr[right] = nn;
            return right;
        }

        typedef std::vector<nhood> KNNGraph;
        KNNGraph graph_;
    };


    class Smart {
    public:
        std::vector<unsigned> each_ep_;
        unsigned n_threads_ = 30;
    };

   
    class VAMANA {
    public:
        float alpha = 1.0;
        unsigned R_refine;
        unsigned L_refine;
        unsigned ep_;
    };


    class HNSW {
    public:
        unsigned NN_ ;
        unsigned ef_construction_ = 100;
        unsigned m_ = 12;
        unsigned max_m_ = 12;
        unsigned max_m0_ = 24;
        int mult;
        float level_mult_ = 1 / log(1.0*m_);

        int max_level_ = 0;
        mutable std::mutex max_level_guard_;

        template <typename KeyType, typename DataType>
        class MinHeap {
        public:
            class Item {
            public:
                KeyType key;
                DataType data;
                Item() {}
                Item(const KeyType& key) :key(key) {}
                Item(const KeyType& key, const DataType& data) :key(key), data(data) {}
                bool operator<(const Item& i2) const {
                    return key > i2.key;
                }
            };

            MinHeap() {
            }

            const KeyType top_key() {
                if (v_.size() <= 0) return 0.0;
                return v_[0].key;
            }

            Item top() {
                if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
                return v_[0];
            }

            void pop() {
                std::pop_heap(v_.begin(), v_.end());
                v_.pop_back();
            }

            void push(const KeyType& key, const DataType& data) {
                v_.emplace_back(Item(key, data));
                std::push_heap(v_.begin(), v_.end());
            }

            size_t size() {
                return v_.size();
            }

        private:
            std::vector<Item> v_;
        };

        class HnswNode {
        public:
            explicit HnswNode(int id, int level, size_t max_m, size_t max_m0)
                    : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level+1) {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].reserve(max_m_ + 1);

                friends_at_layer_[0].reserve(max_m0_ + 1);
            }

            HnswNode(int id, int level, size_t max_m, size_t max_m0, TYPE init_type)
                    : id_(id), level_(level), max_m_(max_m), max_m0_(max_m0), friends_at_layer_(level+1) {
                for (int i = 1; i <= level; ++i)
                    friends_at_layer_[i].resize(max_m_ + 1, nullptr);

                friends_at_layer_[0].resize(max_m0_ + 1, nullptr);
            }

            inline int GetId() const { return id_; }
            inline void SetId(int id) {id_ = id; }
            inline int GetLevel() const { return level_; }
            inline void SetLevel(int level) {level_ = level; }
            inline size_t GetMaxM() const { return max_m_; }
            inline size_t GetMaxM0() const { return max_m0_; }

            inline std::vector<HnswNode*>& GetFriends(int level) { return friends_at_layer_[level]; }
            inline void SetFriends(int level, std::vector<HnswNode*>& new_friends) {
                if (level >= friends_at_layer_.size())
                friends_at_layer_.resize(level + 1);

                friends_at_layer_[level].swap(new_friends);
            }
            inline std::mutex& GetAccessGuard() { return access_guard_; }

            // 1. The list of friends is sorted
            // 2. bCheckForDup == true addFriend checks for duplicates using binary searching
            inline void AddFriends(HnswNode* element, bool bCheckForDup) {
                std::unique_lock<std::mutex> lock(access_guard_);

                if(bCheckForDup) {
                    auto it = std::lower_bound(friends_at_layer_[0].begin(), friends_at_layer_[0].end(), element);
                    if(it == friends_at_layer_[0].end() || (*it) != element) {
                        friends_at_layer_[0].insert(it, element);
                    }
                }else{
                    friends_at_layer_[0].push_back(element);
                }
            }

            // 计算第level层，对field部分已经存了多少个neighbor
            inline unsigned countNeighbor(int level, unsigned starPos, unsigned endPos) {
                unsigned pos = starPos;
                for (; pos < endPos; pos++) {
                    if (friends_at_layer_[level][pos] == nullptr) { break; }
                }
                return pos - starPos;
            }

            inline unsigned countFriends_whole(int level) {
                unsigned count = 0;
                std::vector<HnswNode*> &friends = friends_at_layer_[level];
                for (unsigned pos = 0; pos < friends.size(); pos++) {
                    if (friends[pos] != nullptr) { count++; }
                }
                return count;
            }


            inline bool hasFriend_multi(unsigned id, int level, unsigned starPos, unsigned endPos) {
                std::vector<HnswNode*> &friends = friends_at_layer_[level];
                for (; starPos < endPos; starPos++) {
                    if (friends[starPos] == nullptr) { return false; }
                    if (friends[starPos]->GetId() == id) { return true; }
                }
                return false;
            }


        private:
            int id_;
            int level_;
            size_t max_m_;
            size_t max_m0_;

            std::vector<std::vector<HnswNode*> > friends_at_layer_;
            std::mutex access_guard_;
        };

        class FurtherFirst {
        public:
            FurtherFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode* GetNode() const { return node_; }
            bool operator< (const FurtherFirst& n) const {
                return (distance_ < n.GetDistance());
            }
        protected:
            HnswNode* node_;
            float distance_;
        };

        class CloserFirst {
        public:
            CloserFirst(HnswNode* node, float distance) : node_(node), distance_(distance) {}
            inline float GetDistance() const { return distance_; }
            inline HnswNode* GetNode() const { return node_; }
            bool operator< (const CloserFirst& n) const {
                return (distance_ > n.GetDistance());
            }
        private:
            HnswNode* node_;
            float distance_;
        };

        class VisitedList {
        public:
            VisitedList(unsigned size) : size_(size), mark_(1) {
                visited_ = new unsigned int[size_];
                memset(visited_, 0, sizeof(unsigned int) * size_);
            }

            ~VisitedList() { delete[] visited_; }

            inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }

            inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }

            inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }

            inline void Reset() {
                if (++mark_ == 0) {
                    mark_ = 1;
                    memset(visited_, 0, sizeof(unsigned int) * size_);
                }
            }

            inline unsigned int *GetVisited() { return visited_; }

            inline unsigned int GetVisitMark() { return mark_; }

        private:
            unsigned int *visited_;
            unsigned int size_;
            unsigned int mark_;
        };

        HnswNode* enterpoint_ = nullptr;
        std::vector<HnswNode*> nodes_;
    };










    class MultiIndex : public NNDescent, public Smart, public VAMANA, public HNSW
    {
    public:
        explicit MultiIndex() {
            dist_ = new Distance();
        }

        ~MultiIndex() {
            delete dist_;
        }

        struct SimpleNeighbor{
            unsigned id;
            float distance;

            SimpleNeighbor() = default;
            SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

            inline bool operator<(const SimpleNeighbor &other) const {
                return distance < other.distance;               
            }
        };

        struct SimpleNeighborMultiGroupInfo{
            unsigned id_;
            std::vector<float> repre_dist_;
            std::vector<float> each_field_dist_;

            SimpleNeighborMultiGroupInfo() = default;

            SimpleNeighborMultiGroupInfo(unsigned id, const std::vector<float> &repre_dist, const std::vector<float> &each_field_dist) : id_{id} {
                repre_dist_.resize(repre_dist.size());
                each_field_dist_.resize(each_field_dist.size());
                for (int i = 0; i < repre_dist.size(); i++) { repre_dist_[i] = repre_dist[i]; }
                for (int i = 0; i < each_field_dist.size(); i++) { each_field_dist_[i] = each_field_dist[i]; }
            }

            void setAllDistInfo(const std::vector<float> &repre_dist, const std::vector<float> &each_field_dist)
            {
                repre_dist_.resize(repre_dist.size());
                each_field_dist_.resize(each_field_dist.size());
                for (int i = 0; i < repre_dist.size(); i++) { repre_dist_[i] = repre_dist[i]; }
                for (int i = 0; i < each_field_dist.size(); i++) { each_field_dist_[i] = each_field_dist[i]; }
            }
        };

        float *getBaseData(int field) const {
            return base_data_list_[field];
        }
        
        const std::vector<float *> &getBaseDataList() const {
            return base_data_list_;
        }

        void emplaceBaseData(float *baseData) {
            base_data_list_.emplace_back(baseData);
        }

        void clearBaseData() {
            for (auto base_data_ : base_data_list_) {
                delete[] base_data_;
                base_data_ = nullptr;
            }
        }

        float *getQueryData(int field) const {
            return query_data_list_[field];
        }

        const std::vector<float *> &getQueryDataList() const {
            return query_data_list_;
        }

        void emplaceQueryData(float *queryData) {
            query_data_list_.emplace_back(queryData);
        }

        void clearQueryData() {
            for (auto query_data_ : query_data_list_) {
                delete[] query_data_;
                query_data_ = nullptr;
            }
        }

        unsigned int *getGroundData() const {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData) {
            ground_data_ = groundData;
        }

        void reSetGroundData(unsigned int *groundData, unsigned K) {
            if (ground_data_ != nullptr) {
                delete[] ground_data_;
            }
            ground_data_ = groundData;
            setGroundDim(K);
        }

        unsigned int getBaseLen() const {
            return base_len_;
        }

        void setBaseLen(unsigned int baseLen) {
            base_len_ = baseLen;
        }

        unsigned int getQueryLen() const {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen) {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen) {
            ground_len_ = groundLen;
        }

        std::vector<unsigned> &getBaseDimList() {
            return base_dim_list_;
        }

        unsigned int getBaseDim(int field) const {
            return base_dim_list_[field];
        }

        void emplaceBaseDim(unsigned int baseDim) {
            base_dim_list_.emplace_back(baseDim);
        }

        void setTotalBaseDim() {
            base_total_dim_ = 0;
            for (auto dim: base_dim_list_) {
                base_total_dim_ += dim;
            }
        }

        unsigned getTotalBaseDim() {
            return base_total_dim_;
        }

        std::vector<unsigned> getQueryDimList() const {
            return query_dim_list_;
        }

        unsigned int getQueryDim(int field) const {
            return query_dim_list_[field];
        }

        void emplaceQueryDim(unsigned int queryDim) {
            query_dim_list_.emplace_back(queryDim);
        }

        unsigned int getGroundDim() const {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim) {
            ground_dim_ = groundDim;
        }

        void setSearchWeight(std::vector<std::vector<float>> &search_weight) {
            search_weight_ = std::move(search_weight);
        }

        std::vector<std::vector<float>> getSearchWeight() {
            return search_weight_;
        }

        std::vector<float> &getSearchWeight_q(int query) {
            return search_weight_[query];
        }

        Parameters &getParam() {
            return param_;
        }

        void setParam(const Parameters &param) {
            param_ = param;
            if (param.exist("search_rela_sim_thresh")) {
                search_rela_sim_thresh_ = param.get<float>("search_rela_sim_thresh");
            }
        }

        void setLRefineList(std::vector<unsigned> LRefineList) {
            L_refine_list_ = LRefineList;
        }

        std::vector<unsigned> &getLRefineList() {
            return L_refine_list_;
        }

        void setRRefineList(std::vector<unsigned> RRefineList) {
            R_refine_list_ = RRefineList;
        }

        std::vector<unsigned> &getRRefineList() {
            return R_refine_list_;
        }

        void setAlpha2(float a2) {
            alpha2_ = a2;
        }

        float getAlpha2() {
            return alpha2_;
        }

        void setAlphaNow(float a) {
            alpha_now_ = a;
        }

        float getAlphaNow() {
            return alpha_now_;
        }

        float getSearchRelaSimThresh() {
            return search_rela_sim_thresh_;
        }

        void addRefineRound() {
            refine_round_++;
        }

        unsigned getRefineRound() {
            return refine_round_;
        }

        Distance *getDist() const {
            return dist_;
        }

        void setDist(Distance *dist) {
            dist_ = dist;
        }

        // sorted
        typedef std::vector<std::vector<SimpleNeighbor> > FinalGraph;
        typedef std::vector<std::vector<unsigned> > LoadGraph;
        typedef std::vector<std::vector<SimpleNeighborMultiGroupInfo>> ProcessingGraph;

        // ProcessingGraph
        std::vector<ProcessingGraph> &getProcessingGraphList() {
            return processing_graph_list_;
        }

        ProcessingGraph &getProcessingGraph(int group) {
            return processing_graph_list_[group];
        }

        void clearProcessingGraph(int group) {
            std::vector<std::vector<SimpleNeighborMultiGroupInfo>>().swap(processing_graph_list_[group]);
        }

        // FinalGraph
        std::vector<FinalGraph> &getFinalGraphList() {
            return final_graph_list_;
        }

        FinalGraph &getFinalGraph(int group) {
            return final_graph_list_[group];
        }

        std::vector<xmt::MultiIndex::SimpleNeighbor> &getOutNeigh(int group, unsigned id) {
            return final_graph_list_[group][id];
        }

        void clearFinalGraph(int group) {
            std::vector<std::vector<SimpleNeighbor>>().swap(final_graph_list_[group]);
        }

        void compactNewFinalGraphList() 
        {
            for (int g = 0; g < final_graph_list_.size(); g++) {
                auto &graph_ = final_graph_list_[g];
#pragma omp parallel
                {
                    const int R = R_refine_list_[g];
#pragma omp for schedule(dynamic, 100)
                    for (unsigned i = 0; i < base_len_; i++) {
                        int tmp_size;
                        auto &n_ON = graph_[i];
                        for (tmp_size = 0; tmp_size < R; tmp_size++) {
                            if (n_ON[tmp_size].distance < 0.0f) {
                                break;
                            }
                        }
                        if (tmp_size < R) { n_ON.resize(tmp_size); }
                    }
                }
            }
        }

        void compactFinalGraph(int g) 
        {
            auto &graph_ = final_graph_list_[g];
#pragma omp parallel
            {
                const int R = R_refine_list_[g];
#pragma omp for schedule(dynamic, 100)
                for (unsigned i = 0; i < base_len_; i++) {
                    int tmp_size;
                    auto &n_ON = graph_[i];
                    for (tmp_size = 0; tmp_size < R; tmp_size++) {
                        if (n_ON[tmp_size].distance < 0.0f) {
                            break;
                        }
                    }
                    if (tmp_size < R) { n_ON.resize(tmp_size); }
                }
            }
        }

        void makeSureFinalGraphSpace(int group) {
            int R = R_refine_list_[group];
            auto &graph_ = final_graph_list_[group];
#pragma omp parallel
            {
                int tmp_size;
#pragma omp for schedule(dynamic, 100)
                for (unsigned i = 0; i < base_len_; i++) {
                    auto &n_ON = graph_[i];
                    tmp_size = n_ON.size();
                    if (tmp_size < R) { 
                        n_ON.resize(R); 
                        for (int i = tmp_size; i < R; i++) {
                            n_ON[i].distance = -1;
                        }
                    }
                    
                }
            }
        }

        std::vector<LoadGraph> &getLoadGraphList() {
            return load_graph_list_;
        }
        
        LoadGraph &getLoadGraph(int group) {
            return load_graph_list_[group];
        }

        void clearLoadGraph(int group) {
            std::vector<std::vector<unsigned>>().swap(load_graph_list_[group]);
        }

        std::vector<LoadGraph> &getExactGraphList() {
            return exact_graph_list_;
        }

        LoadGraph &getExactGraph(int group) {
            return exact_graph_list_[group];
        }

        void clearExactGraph(int group) {
            std::vector<std::vector<unsigned>>().swap(exact_graph_list_[group]);
        }

        unsigned int getDistCount() const {
            return dist_count;
        }

        void resetDistCount() {
            dist_count = 0;
        }

        void addDistCount() {
            dist_count += 1;
        }

        unsigned int getHopCount() const {
            return hop_count;
        }

        void resetHopCount() {
            hop_count = 0;
        }

        void addHopCount() {
            hop_count += 1;
        }
        
        void setNumThreads(const unsigned numthreads) {
            omp_set_num_threads(numthreads);
            std::cout << "\n ======================================" << std::endl;
            std::cout << " = omp_set_num_threads(" << numthreads << ")" << std::endl;
            std::cout << " ======================================\n" << std::endl;
        }

        void setFieldNum(unsigned num_field) {
            field_num_ = num_field;
        }

        unsigned getFieldNum() {
            return field_num_;
        }

        void setGroupList(std::vector<std::vector<int>> GroupFieldList) {
            group_field_list_ = GroupFieldList;
        }

        std::vector<std::vector<int>> &getGroupList() {
            return group_field_list_;
        }

        std::vector<int> &getGroupField(int g) {
            return group_field_list_[g];
        }

        unsigned getGroupNum() {
            return std::max(group_field_list_.size(), group_represent_list_.size());
        }

        void setGroupRepreList(std::vector<std::vector<float>> GroupRepreList) {
            group_represent_list_ = GroupRepreList;
        }

        std::vector<std::vector<float>> &getGroupRepreList() {
            return group_represent_list_;
        }

        std::vector<float> &getGroupRepre(int g) {
            return group_represent_list_[g];
        }

        void setRelaCheckList(std::vector<std::vector<int>> RelatedCheckList) {
            related_check_list_ = RelatedCheckList;
        }

        std::vector<std::vector<int>> &getRelaCheckList_modify() {
            return related_check_list_;
        }

        const std::vector<std::vector<int>> &getRelaCheckList() const {
            return related_check_list_;
        }

        const std::vector<int> &getRelaCheck(int group) const { // 第group组index在构建/search时需要查看的相关index
            return related_check_list_[group];
        }

        void summaryLoadGraph(std::vector<int> groupList, std::vector<unsigned> idList = {}) {
            if (idList.size() == 0) {
                idList.emplace_back(0);
                if (base_len_ >= 1) {
                    idList.emplace_back(1);
                }
                if (base_len_ >= 2) {
                    idList.emplace_back(base_len_-1);
                }
            }
            std::cout << "######### Summary: load_graph_list_ of smart_builder #########" << std::endl;
            if (getLoadGraph(groupList[0]).size()) {
                // ## 1. 展示groupList中提及到的index，每个index都各自平均有几个out-neighbors
                for (int g : groupList) 
                {
                    float ON = 0.0;
                    unsigned maxON = 0, whose;
                    for (unsigned id = 0; id < base_len_; id++) {
                        ON += getLoadGraph(g)[id].size();
                        if (getLoadGraph(g)[id].size() > maxON) {
                            whose = id;
                            maxON = getLoadGraph(g)[id].size();
                        }
                    }
                    ON /= (0.0 + base_len_);
                    std::cout << "-- Average out-neighbors per item of Index[" << g << "]: " << ON << std::endl;
                    std::cout << "           max out-neighbors is Index[" << g << "][" << whose << "]: " << maxON << std::endl;
                    std::cout << "              Neighbors for " << whose << ":  ";
                    for (int neigh = 0; neigh < getLoadGraph(g)[whose].size(); neigh++) {
                        std::cout << getLoadGraph(g)[whose][neigh] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "________________________________________________________\n" <<  std::endl;

                // ## 2. 展示几个out-neighbor情况
                for (unsigned id : idList) {
                    std::cout << "- For " << id << ":\n";
                    for (int g : groupList) {
                        std::cout << "-- For field group " << g << " (with field: ";
                        for (int f : getGroupList()[g]) { std::cout << f << ", "; }
                        std::cout << ")\nNeighbors for " << id << ":\n       ";
                        for (int neigh = 0; neigh < getLoadGraph(g)[id].size(); neigh++) {
                            std::cout << getLoadGraph(g)[id][neigh] << ", ";
                        }
                        std::cout << std::endl;
                        std::cout << "--------------------------------------------" << std::endl;
                    }
                    std::cout << "==================================================\n" << std::endl;     
                }
                std::cout << "##################################################################" << std::endl;
            } else {
                std::cout << "\n!!!! The load_graph_list_ is empty !!!!" << std::endl;
                std::cout << "##################################################################" << std::endl;
            } 
        }

        void summaryFinalGraph(std::vector<int> groupList, std::vector<unsigned> idList = {}) {
            if (idList.size() == 0) {
                idList.emplace_back(0);
                if (base_len_ >= 1) {
                    idList.emplace_back(1);
                }
                if (base_len_ >= 2) {
                    idList.emplace_back(base_len_-1);
                }
            }
            std::cout << "######### Summary: final_graph_list of smart_builder #########" << std::endl;
            if (getFinalGraph(groupList[0]).size()) {
                // ## 1. 展示groupList中提及到的index，每个index都各自平均有几个out-neighbors
                for (int g : groupList) 
                {
                    float ON = 0.0;
                    unsigned maxON = 0, whose;
                    for (unsigned id = 0; id < base_len_; id++) {
                        ON += getFinalGraph(g)[id].size();
                        if (getFinalGraph(g)[id].size() > maxON) {
                            whose = id;
                            maxON = getFinalGraph(g)[id].size();
                        }
                    }
                    ON /= (0.0 + base_len_);
                    std::cout << "-- Average out-neighbors per item of Index " << g << ": " << ON << std::endl;
                    std::cout << "           max out-neighbors is Index[" << g << "][" << whose << "]: " << maxON << std::endl;
                    std::cout << "                  Neighbors for " << whose << ":  ";
                    for (int neigh = 0; neigh < getFinalGraph(g)[whose].size(); neigh++) {
                        std::cout << getFinalGraph(g)[whose][neigh].id << ", ";
                    }
                    std::cout << std::endl;
                    std::cout << "                  with distance: " << ": ";
                    for (int neigh = 0; neigh < getFinalGraph(g)[whose].size(); neigh++) {
                        std::cout << getFinalGraph(g)[whose][neigh].distance << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "________________________________________________________\n" <<  std::endl;

                

                // ## 2. 展示几个out-neighbor情况
                for (unsigned id : idList) {
                    std::cout << "- For " << id << ":\n";
                    for (int g : groupList) {
                        std::cout << "-- For field group " << g << " (with field: ";
                        for (int f : getGroupList()[g]) { std::cout << f << ", "; }
                        std::cout << ")\nNeighbors for " << id << ":\n       ";
                        for (int neigh = 0; neigh < getFinalGraph(g)[id].size(); neigh++) {
                            std::cout << getFinalGraph(g)[id][neigh].id << ", ";
                        }
                        std::cout << std::endl;
                        std::cout << " with distance: " << ":\n       ";
                        for (int neigh = 0; neigh < getFinalGraph(g)[id].size(); neigh++) {
                            std::cout << getFinalGraph(g)[id][neigh].distance << ", ";
                        }
                        std::cout << std::endl;
                        std::cout << "--------------------------------------------" << std::endl;
                    }
                    std::cout << "==================================================\n" << std::endl;     
                }
                std::cout << "##################################################################" << std::endl;
            } else {
                std::cout << "\n!!!! The final_graph_ is empty !!!!" << std::endl;
                std::cout << "##################################################################" << std::endl;
            } 
        }

        bool checkFinalGraph(int group) {
            bool correct = true;
            const auto &graph_ = final_graph_list_[group];
#pragma omp for schedule(dynamic, 100)
            for (unsigned id = 0; id < base_len_; id++) {
                for (const auto &ON :  graph_[id]) {
                    if (ON.distance < 0.0f) {
                        std::cout << "-- Has out-neighbor of distance -1.0: id = " << id << std::endl;
                        correct = false;
                        break;
                    }
                }
            }
            return correct;
        }

        void copyOldGraphList() {
            processing_graph_list_.resize(final_graph_list_.size());
            old_final_graph_list_.resize(final_graph_list_.size());
            for (int ind = 0; ind < final_graph_list_.size(); ind++) {
                old_final_graph_list_[ind].resize(final_graph_list_[ind].size());
                processing_graph_list_[ind].resize(final_graph_list_[ind].size());
                for (unsigned id = 0; id < final_graph_list_[ind].size(); id++) {
                    old_final_graph_list_[ind][id].resize(final_graph_list_[ind][id].size());
                    for (unsigned nn = 0; nn < old_final_graph_list_[ind][id].size(); nn++) {
                        old_final_graph_list_[ind][id][nn] = final_graph_list_[ind][id][nn];
                    }
                }
            }
        }

        void clearOldGraphList() {
            std::vector<FinalGraph>().swap(old_final_graph_list_);
        }

        std::vector<FinalGraph> &getOldGraphList() {
            return old_final_graph_list_;
        }

        FinalGraph &getOldGraph(int group) {
            return old_final_graph_list_[group];
        }

        std::vector<xmt::MultiIndex::SimpleNeighbor> &getOldOutNeigh(int group, unsigned id) {
            return old_final_graph_list_[group][id];
        }

        static inline bool extractBracketIntList(const std::string& line,
                                                const std::string& key,
                                                std::vector<int>& out) {
            auto pos = line.find(key);
            if (pos == std::string::npos) return false;

            auto lb = line.find('[', pos);
            auto rb = line.find(']', pos);
            if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;

            std::string inside = line.substr(lb + 1, rb - lb - 1);
            std::stringstream ss(inside);
            out.clear();
            while (ss.good()) {
                int x;
                char c;
                if (!(ss >> x)) break;
                out.push_back(x);
                ss >> c; 
            }
            return !out.empty();
        }

        static bool parseRepProtoIdsFromLog(const std::string& log_path,
                                        std::vector<int>& rep_proto_ids) {
            std::ifstream fin(log_path);
            if (!fin) return false;

            rep_proto_ids.clear();

            std::string line;
            // 1) Prefer "Final order: [...]"
            while (std::getline(fin, line)) {
                std::vector<int> tmp;
                if (extractBracketIntList(line, "Final order:", tmp)) {
                    // 要用 txt 保存的顺序，而它通常是 Final order 的 reverse
                    std::reverse(tmp.begin(), tmp.end());
                    rep_proto_ids = std::move(tmp);
                    return true;
                }
            }

            // 2) Fallback: parse "prototype: Query X"
            fin.clear();
            fin.seekg(0);
            while (std::getline(fin, line)) {
                // prototype: Query 44 → [...]
                auto p = line.find("prototype:");
                if (p == std::string::npos) continue;

                auto qpos = line.find("Query", p);
                if (qpos == std::string::npos) continue;

                qpos += 4;
                while (qpos < line.size() && line[qpos] == ' ') qpos++;

                // read integer
                int id = -1;
                {
                    std::stringstream ss(line.substr(qpos));
                    ss >> id;
                }
                if (id >= 0) rep_proto_ids.push_back(id);
            }
            return !rep_proto_ids.empty();
        }

        bool parseSimRowFromLog(const std::string& log_path,
                                    int target_row_id,
                                    std::vector<float>& corr_row) {
            std::ifstream fin(log_path);
            if (!fin) return false;

            corr_row.clear();

            std::string line;
            bool in_block = false;

            while (std::getline(fin, line)) {
                if (!in_block) {
                    if (line.find("Init sim_matrix") != std::string::npos) {
                        in_block = true;
                    }
                    continue;
                }

                // stop condition: next section starts
                if (!line.empty() && line[0] == '#') break;

                // match: "<row_id>:  1.000000  -0.044406  ..."
                // trim leading spaces
                size_t i = 0;
                while (i < line.size() && line[i] == ' ') i++;

                // parse leading integer
                int row_id = -1;
                size_t colon = line.find(':', i);
                if (colon == std::string::npos) continue;

                {
                    std::string head = line.substr(i, colon - i);
                    std::stringstream ss(head);
                    ss >> row_id;
                }
                if (row_id != target_row_id) continue;

                // parse floats after colon
                std::string rest = line.substr(colon + 1);
                std::stringstream ss(rest);
                float x;
                while (ss >> x) corr_row.push_back(x);

                // 我们期待 pow(2, getFieldNum())-1 个
                return (corr_row.size() >= pow(2, getFieldNum())-1);
            }

            return false;
        }

        bool parseSimRowFromLog(const std::string& log_path,
                                std::vector<float>& weight,
                                std::vector<float>& corr_row) {

            int target_row_id = -2;
            std::ifstream fin(log_path);
            if (!fin) return false;

            corr_row.clear();

            std::string line;
            bool weight_in_block = false;

            while (std::getline(fin, line)) {
                if (!weight_in_block) 
                {
                    if (line.find("Showing All Workload Weight") != std::string::npos) {
                        weight_in_block = true;

                        // 开始继续往下读，直到 Init sim_matrix
                        target_row_id = [&]() -> int {
                            int found_row_id = -1;   // 默认没找到
                            std::string inner_line;

                            while (std::getline(fin, inner_line)) {
                                // 一旦到达 sim_matrix 区域，停止查找 weight
                                if (inner_line.find("Init sim_matrix") != std::string::npos) {
                                    break;
                                }

                                // 跳过空行
                                if (inner_line.empty()) continue;

                                // 解析格式： row_id : [w1, w2, ...]
                                size_t i = 0;
                                while (i < inner_line.size() && inner_line[i] == ' ') i++;

                                size_t colon = inner_line.find(':', i);
                                if (colon == std::string::npos) continue;

                                int row_id = -1;
                                {
                                    std::string head = inner_line.substr(i, colon - i);
                                    std::stringstream ss(head);
                                    if (!(ss >> row_id)) continue;
                                }

                                size_t lb = inner_line.find('[', colon);
                                size_t rb = inner_line.find(']', colon);
                                if (lb == std::string::npos || rb == std::string::npos || rb <= lb) continue;

                                std::string body = inner_line.substr(lb + 1, rb - lb - 1);

                                // 把逗号替换为空格，方便解析 float
                                for (char& c : body) {
                                    if (c == ',') c = ' ';
                                }

                                std::vector<float> parsed_weight;
                                std::stringstream ss(body);
                                float x;
                                while (ss >> x) parsed_weight.push_back(x);

                                // 比较 parsed_weight 和输入 weight 是否相同
                                if (parsed_weight.size() == weight.size()) {
                                    bool same = true;
                                    for (size_t k = 0; k < weight.size(); ++k) {
                                        if (std::fabs(parsed_weight[k] - weight[k]) > 1e-6f) {
                                            same = false;
                                            break;
                                        }
                                    }

                                    if (same) {
                                        found_row_id = row_id;
                                        std::cout << "found_row_id = " << found_row_id << std::endl;
                                    }
                                }
                            }

                            return found_row_id;
                        }();

                        // 如果在 weight block 中没有找到对应 weight
                        if (target_row_id == -1) return false;

                        // 此时文件流已经被上面的 lambda 读到了 Init sim_matrix 后面
                        // 后面直接继续读 sim_matrix 内容
                        continue;
                    }

                    if (line.find("Init sim_matrix") != std::string::npos) {
                        return false;
                    }

                    continue;
                }

                // stop condition: next section starts
                if (!line.empty() && line[0] == '#') break;

                // match: "<row_id>:  1.000000  -0.044406  ..."
                // trim leading spaces
                size_t i = 0;
                while (i < line.size() && line[i] == ' ') i++;

                // parse leading integer
                int row_id = -1;
                size_t colon = line.find(':', i);
                if (colon == std::string::npos) continue;

                {
                    std::string head = line.substr(i, colon - i);
                    std::stringstream ss(head);
                    ss >> row_id;
                }

                if (row_id != target_row_id) continue;

                // parse floats after colon
                std::string rest = line.substr(colon + 1);
                std::stringstream ss(rest);
                float x;
                while (ss >> x) corr_row.push_back(x);

                // 我们期待 pow(2, getFieldNum())-1 个
                return (corr_row.size() >= pow(2, getFieldNum()) - 1);
            }

            return false;
        }

        void preSetNeedIndeces_FallbackIntersect(std::vector<float>& weight, std::string &bash_path) {
            need_indeces_.clear();
            float corr_thresh = getSearchRelaSimThresh();

            // 1) parse representative proto ids (group order)
            std::vector<int> rep_proto_ids;
            if (!parseRepProtoIdsFromLog(bash_path, rep_proto_ids)) {
                std::cout << "[GS-corr] Failed to parse representative proto ids from log: "
                        << bash_path << ". Fallback to Intersect\n";
                exit(-1);
            }

            // 2) parse correlation row for this query_id
            std::vector<float> corr_row;
            if (!parseSimRowFromLog(bash_path, weight, corr_row)) {
                std::cout << "[GS-corr] Failed to parse sim_matrix row for weight [...]" 
                        << " from log: " << bash_path << ". Fallback to Intersect\n";
                // fallback: all groups
                need_indeces_.reserve(rep_proto_ids.size());
                for (int g = 0; g < (int)rep_proto_ids.size(); ++g) need_indeces_.push_back(g);
                return;
            }

            // 3) enable groups with corr >= thresh (group id is index in rep_proto_ids)
            for (int g = 0; g < (int)rep_proto_ids.size(); ++g) {
                const int proto_id = rep_proto_ids[g];
                float corr = -1e9f;
                if (proto_id >= 0 && proto_id < (int)corr_row.size()) corr = corr_row[proto_id];

                if (corr >= corr_thresh) {
                    need_indeces_.push_back(g); // g is "the group id"
                }
            }

            // 4) fallback if none reached
            // Fallback to intersect (just return empty?!)
            if (need_indeces_.empty()) {

                // ### --- 方法二： 简单判断，有交集就使用 ---
                {
                    for (int ind = 0; ind < getGroupNum(); ind++)
                    {
                        auto &fElements = getGroupList()[ind];
                        bool skip = true;
                        for (auto f : fElements)
                        {
                            if (weight[f])
                            {
                                skip = false;
                                break;
                            }
                        }
                        if (!skip)
                        {
                            need_indeces_.emplace_back(ind);
                        }
                    }
                }

                std::cout << "[GS-corr] Not Reached Corr (tau=" << corr_thresh << ", fallback to GS-intersect\n";
            }
        }
        
        std::vector<int> getNeedIndeces() {
            return need_indeces_;
        }

        //  ----- 处理correlation相关内容 -----
        static inline std::string trim_copy(const std::string& s) {
            size_t b = 0, e = s.size();
            while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) b++;
            while (e > b && (s[e-1] == ' ' || s[e-1] == '\t' || s[e-1] == '\r' || s[e-1] == '\n')) e--;
            return s.substr(b, e - b);
        }

        static inline bool starts_with(const std::string& s, const std::string& p) {
            return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
        }

        static inline void parse_floats_after_colon(const std::string& line, std::vector<float>& out) {
            auto pos = line.find(':');
            out.clear();
            if (pos == std::string::npos) return;
            std::stringstream ss(line.substr(pos + 1));
            float x;
            while (ss >> x) out.push_back(x);
        }

        static inline void parse_ints_after_colon(const std::string& line, std::vector<int>& out) {
            auto pos = line.find(':');
            out.clear();
            if (pos == std::string::npos) return;
            std::stringstream ss(line.substr(pos + 1));
            int x;
            while (ss >> x) out.push_back(x);
        }

        static inline bool parse_tau_from_enabled_line(const std::string& line, float& tau_out) {
            // line example: "enabled_groups@tau=0.3: 1 2"
            auto p = line.find("enabled_groups@tau=");
            if (p == std::string::npos) return false;
            p += std::string("enabled_groups@tau=").size();
            auto c = line.find(':', p);
            if (c == std::string::npos) return false;
            std::string tau_str = line.substr(p, c - p);
            try {
                tau_out = std::stof(tau_str);
                return true;
            } catch (...) {
                return false;
            }
        }

        bool loadEnabledGroupsCacheFromTxt(const std::string& cache_txt_path) {
            std::ifstream fin(cache_txt_path);
            if (!fin) {
                std::cout << "[Cache] Cannot open cache txt: " << cache_txt_path << "\n";
                enabled_cache_loaded_ = false;
                return false;
            }

            // temp store
            std::vector<std::vector<float>> weights;
            std::vector<std::vector<int>> enabled_groups;

            int m_from_file = -1;
            float tau_from_file = -1.0f;

            std::string line;
            std::vector<float> cur_vec;
            std::vector<int> cur_enabled;

            bool has_vec = false;
            bool has_enabled = false;

            while (std::getline(fin, line)) {
                line = trim_copy(line);
                if (line.empty()) continue;

                if (starts_with(line, "num_fields:")) {
                    std::stringstream ss(line.substr(std::string("num_fields:").size()));
                    ss >> m_from_file;
                    continue;
                }

                if (starts_with(line, "vec:")) {
                    parse_floats_after_colon(line, cur_vec);
                    has_vec = !cur_vec.empty();
                    continue;
                }

                if (line.find("enabled_groups@tau=") != std::string::npos &&
                    line.find("enabled_groups_with_rela") == std::string::npos) {

                    if (tau_from_file < 0) {
                        float tmp_tau;
                        if (parse_tau_from_enabled_line(line, tmp_tau)) tau_from_file = tmp_tau;
                    }
                    parse_ints_after_colon(line, cur_enabled);
                    has_enabled = !cur_enabled.empty() || true; 
                    continue;
                }

                if (starts_with(line, "Q") && line.find("key=") != std::string::npos) {
                    if (has_vec && has_enabled) {
                        weights.push_back(cur_vec);
                        enabled_groups.push_back(cur_enabled);
                    }
                    cur_vec.clear(); cur_enabled.clear();
                    has_vec = false; has_enabled = false;
                    continue;
                }
            }
            if (has_vec && has_enabled) {
                weights.push_back(cur_vec);
                enabled_groups.push_back(cur_enabled);
            }

            if (m_from_file <= 0 || weights.empty()) {
                std::cout << "[Cache] Parse failed: m=" << m_from_file << ", tau=" << tau_from_file
                        << ", weights=" << weights.size() << "\n";
                enabled_cache_loaded_ = false;
                return false;
            }

            for (size_t i = 0; i < weights.size(); ++i) {
                if ((int)weights[i].size() != m_from_file) {
                    std::cout << "[Cache] Bad weight dim at i=" << i << " got=" << weights[i].size()
                            << " expected=" << m_from_file << "\n";
                    enabled_cache_loaded_ = false;
                    return false;
                }
            }

            // build flat storage
            const uint32_t W = (uint32_t)weights.size();
            cached_tau_ = tau_from_file;
            assert(field_num_ == (unsigned)m_from_file);
            unique_weight_num_ = W;

            cached_weights_flat_.assign(W * field_num_, 0.0f);
            for (uint32_t w = 0; w < W; ++w) {
                for (unsigned j = 0; j < field_num_; ++j) {
                    cached_weights_flat_[w * field_num_ + j] = weights[w][j];
                }
            }

            cached_enabled_offsets_.assign(W + 1, 0);
            uint32_t total = 0;
            for (uint32_t w = 0; w < W; ++w) {
                cached_enabled_offsets_[w] = total;
                total += (uint32_t)enabled_groups[w].size();
            }
            cached_enabled_offsets_[W] = total;

            cached_enabled_flat_.assign(total, 0);
            uint32_t ptr = 0;
            for (uint32_t w = 0; w < W; ++w) {
                for (int g : enabled_groups[w]) {
                    cached_enabled_flat_[ptr++] = (uint16_t)g;
                }
            }

            enabled_cache_loaded_ = true;

            std::cout << "[Cache] Loaded enabled_groups cache: W=" << W
                    << ", m=" << field_num_
                    << ", cached_tau=" << cached_tau_ << "\n";

            std::cout << "[cached_enabled_offsets_]: ";
            for (auto f : cached_enabled_offsets_) {std::cout << f << ", ";}
            std::cout << std::endl;

            std::cout << "[unique_weight,   num_related(cached_enabled_offsets_[W+1] - cached_enabled_offsets_[W]),  related_groups]" << std::endl;
            for (uint32_t w = 0; w < W; ++w) {
                std::cout << "For uniq weight-" << w << ":   [";
                for (unsigned j = w*field_num_; j < w*field_num_+field_num_; ++j) {
                    std::cout << cached_weights_flat_[j] << ", ";
                }
                std::cout << "],   has " << (cached_enabled_offsets_[w+1] - cached_enabled_offsets_[w]) << " enabled groups of [";
                for (unsigned j = cached_enabled_offsets_[w]; j < cached_enabled_offsets_[w+1]; ++j) {
                    std::cout << cached_enabled_flat_[j] << ", ";
                }
                std::cout << "]" << std::endl;
            }
            return true;
        }

        void showSavedEnabledSummary() {
            std::cout << "------ [SavedEnabledSummary] START ---------------" << std::endl;
            std::cout << "num of unique candidate query weight: W=" << unique_weight_num_
                    << ", m=" << field_num_
                    << ", cached_tau=" << cached_tau_ << "\n";

            std::cout << "[cached_enabled_offsets_]: ";
            for (auto f : cached_enabled_offsets_) {std::cout << f << ", ";}
            std::cout << std::endl;

            std::cout << "[cached_enabled_flat_]: ";
            for (auto f : cached_enabled_flat_) {std::cout << f << ", ";}
            std::cout << std::endl;

            std::cout << "[unique_weight,   num_related(cached_enabled_offsets_[w+1] - cached_enabled_offsets_[w]),  related_groups]" << std::endl;
            std::cout << "unique_weight_num_ = " << unique_weight_num_ << std::endl;
            for (uint32_t w = 0; w < unique_weight_num_; ++w) {
                std::cout << "For uniq weight-" << w << ":   [";
                for (unsigned j = w*field_num_; j < w*field_num_+field_num_; ++j) {
                    std::cout << cached_weights_flat_[j] << ", ";
                }
                std::cout << "],   has " << (cached_enabled_offsets_[w+1] - cached_enabled_offsets_[w]) << " enabled groups of [";
                for (unsigned j = cached_enabled_offsets_[w]; j < cached_enabled_offsets_[w+1]; ++j) {
                    std::cout << cached_enabled_flat_[j] << ", ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "------ [SavedEnabledSummary] END ---------------" << std::endl;

        }


        float &getRelaThresh(){
            return cached_tau_;
        }

        void setRelaThresh(float t){
            cached_tau_ = t;
        }

        unsigned &getUniqueWorkloadNum(){
            return unique_weight_num_;
        }

        void setUniqueWorkloadNum(unsigned uniq_num){
            unique_weight_num_ = uniq_num;
        }
        
        std::vector<float> &getUniqueWorkloadFlat() {
            return cached_weights_flat_;          // size = W * m1
        }

        void setUniqueWorkloadFlat(std::vector<float> &x) {
            cached_weights_flat_ = x;          // size = W * m1
        }

        std::vector<uint32_t> &getEnableGroupRecordOffset(){
            return cached_enabled_offsets_;    // size = W+1
        }

        void setEnableGroupRecordOffset(std::vector<uint32_t> &x){
            cached_enabled_offsets_ = x;    // size = W+1
        }
        
        std::vector<uint16_t> &getEnabledGroupFlat() {
            return cached_enabled_flat_; 
        }

        void setEnabledGroupFlat(std::vector<uint16_t> &x) {
            cached_enabled_flat_ = x; 
        }

        int i = 0;
        // bool debug = false;

    private:
        std::vector<float*> base_data_list_, query_data_list_;
        unsigned *ground_data_;
        unsigned base_len_, query_len_, ground_len_;
        unsigned base_total_dim_, ground_dim_;
        std::vector<unsigned> base_dim_list_, query_dim_list_;
        std::vector<std::vector<float>> search_weight_;

        Parameters param_;
        std::vector<unsigned> L_refine_list_, R_refine_list_;
        float alpha2_ = 1.0, alpha_now_;
        unsigned refine_round_ = 0;

        Distance *dist_;

        // -- Graph Structures:
        std::vector<ProcessingGraph> processing_graph_list_;
        std::vector<FinalGraph> final_graph_list_;
        std::vector<FinalGraph> old_final_graph_list_;
        std::vector<LoadGraph> load_graph_list_;
        std::vector<LoadGraph> exact_graph_list_;

        unsigned field_num_, group_num_;
        std::vector<std::vector<int>> group_field_list_;
        std::vector<std::vector<float>> group_represent_list_;
        std::vector<std::vector<int>> related_check_list_;
        
        unsigned dist_count = 0;
        unsigned hop_count = 0;
        
        // -- WAGS
        std::vector<int> need_indeces_;
        float search_rela_sim_thresh_;
        // -- about edge-group correlation --
        // -- cache meta
        float cached_tau_ = -1.0f;
        unsigned unique_weight_num_ = 0;
        bool enabled_cache_loaded_ = false;
        // -- cached weights: W x m
        std::vector<float> cached_weights_flat_;          // size = W * m1
        std::vector<uint32_t> cached_enabled_offsets_;    // size = W+1
        std::vector<uint16_t> cached_enabled_flat_;       // store group ids (<=65535 is enough) // enabled group （平铺）
    };

}

#endif //XMT_INDEX_H
