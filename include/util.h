#ifndef XMT_UTIL_H
#define XMT_UTIL_H

#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <iostream>
#include <string>

namespace xmt {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    static void GenRandom_WithReplacement(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % N;
        }
    }

    template<typename T>
    struct Candidate2 {
            size_t row_id;
            T distance;
            Candidate2(const size_t row_id, const T distance): row_id(row_id), distance(distance) { }

            bool operator >(const Candidate2& rhs) const {
                if (this->distance == rhs.distance) {
                    return this->row_id > rhs.row_id;
                }
                return this->distance > rhs.distance;
            }
        };

        typedef std::set<Candidate2<float>, std::greater<Candidate2<float> > > CandidateHeap2;

    
    inline bool dir_exists(const std::string &path) {
        struct stat info;

        if (stat(path.c_str(), &info) != 0) {
            return false;
        }

        return (info.st_mode & S_IFDIR) != 0;
    }

    inline bool create_dir_if_not_exists(const std::string &path) {
        if (dir_exists(path)) {
            return true;
        }

        if (mkdir(path.c_str(), 0755) != 0) {
            if (errno == EEXIST) {
                return true;
            }

            std::cerr << "Error creating directory: " << path << std::endl;
            return false;
        }

        return true;
    }
    
    // --- 用于存ground truth ---- 
    void inline save_ivecs(const std::string &filename, const unsigned int *ground_truth, unsigned query_len, unsigned K) {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            exit(-1);
        }

        for (unsigned i = 0; i < query_len; i++) {
            out.write(reinterpret_cast<const char *>(&K), sizeof(unsigned)); // ivecs 格式：每个向量前先写入维度 K
            out.write(reinterpret_cast<const char *>(ground_truth + i * K),  K * sizeof(unsigned)); // 写入第 i 个 query 的 K 个 ground truth id
        }

        out.close();
        std::cout << "Successfully saved to " << filename << std::endl;
    }
}

#endif //XMT_UTIL_H
