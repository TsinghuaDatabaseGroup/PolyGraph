#ifndef XMT_UTIL_H
#define XMT_UTIL_H

#include <random>
#include <algorithm>

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
}

#endif //XMT_UTIL_H
