#ifndef XMT_DISTANCE_H
#define XMT_DISTANCE_H

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <unordered_set>

#include "policy.h"


struct Timer
{
    std::chrono::high_resolution_clock::time_point start;
    float total = 0.0;

    void tic()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void toc()
    {
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<float, std::micro>(end - start).count(); // 微秒
    }
};

namespace xmt
{
    class Distance
    {
    public:
        template <typename T>
        T compare(const T *a, const T *b, unsigned length, TYPE dist_type = TYPE::DIST_EUCLIDEAN) const
        {
            T dist = 0;

            if (dist_type == TYPE::DIST_COS)
            {
                return compare_cosDist(a, b, length);
            }
            else if (dist_type == TYPE::DIST_COS_SIMILARITY)
            {
                return compare_cosSim(a, b, length);
            }
            else if (dist_type == TYPE::DIST_EUCLIDEAN)
            {
                float diff0, diff1, diff2, diff3;
                const T *last = a + length;
                const T *unroll_group = last - 3;

                /* Process 4 items with each loop for efficiency. */
                while (a < unroll_group)
                {
                    diff0 = a[0] - b[0];
                    diff1 = a[1] - b[1];
                    diff2 = a[2] - b[2];
                    diff3 = a[3] - b[3];
                    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                    a += 4;
                    b += 4;
                }
                /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
                while (a < last)
                {
                    diff0 = *a++ - *b++;
                    dist += diff0 * diff0;
                }

                dist = std::sqrt(dist); // 开关 of weighted square
                return dist;
            }
            else
            {
                std::cout << "Unknown dist_type!" << std::endl;
                exit(-1);
            }
        }

        template <typename T>
        T compare_cosSim(const T *a, const T *b, unsigned length) const
        {
            T result = 0, ab = 0, aa = 0, bb = 0;

            float ab0, ab1, ab2, ab3, aa0, aa1, aa2, aa3, bb0, bb1, bb2, bb3;
            const T *last = a + length;
            const T *unroll_group = last - 3;

            /* Process 4 items with each loop for efficiency. */
            while (a < unroll_group)
            {
                ab0 = a[0] * b[0];
                ab1 = a[1] * b[1];
                ab2 = a[2] * b[2];
                ab3 = a[3] * b[3];
                aa0 = a[0] * a[0];
                aa1 = a[1] * a[1];
                aa2 = a[2] * a[2];
                aa3 = a[3] * a[3];
                bb0 = b[0] * b[0];
                bb1 = b[1] * b[1];
                bb2 = b[2] * b[2];
                bb3 = b[3] * b[3];
                ab += ab0 + ab1 + ab2 + ab3;
                aa += aa0 + aa1 + aa2 + aa3;
                bb += bb0 + bb1 + bb2 + bb3;
                a += 4;
                b += 4;
            }
            /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
            while (a < last)
            {
                aa0 = a[0] * a[0];
                bb0 = b[0] * b[0];
                ab0 = *a++ * *b++;
                aa += aa0;
                ab += ab0;
                bb += bb0;
            }
            result = ab / (sqrt(aa) * sqrt(bb));

            return result;
        }

        template <typename T>
        T compare_cosDist(const T *a, const T *b, unsigned length) const
        {
            T result = 0;
            result = 1.0 - compare_cosSim(a, b, length);
            if (result < 0)
                result = 0;
            else if (result > 2)
                result = 2;
            return result;
        }

        /** 
         * compare_multi():
         *      计算(base1中的)id1 和 (base2中的)id2 在field-group “group” 下的距离
         * @param base1 base1的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id1 base1 下的 ID
         * @param base2 base2的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id2 base2 下的 ID
         * @param dim_list
         * @param need_calcu field-group中涉及到的field编号
         * @param dist_type 距离计算方式
         */
        float compare_multi(std::vector<float *> base1, unsigned id1,
                            std::vector<float *> base2, unsigned id2,
                            std::vector<unsigned> dim_list, std::vector<int> need_calcu,
                            TYPE dist_type = TYPE::DIST_EUCLIDEAN) const
        {
            float result = 0.0;
            for (auto field : need_calcu)
            {
                unsigned dim = dim_list[field];
                const float *a = base1[field] + id1 * dim;
                const float *b = base2[field] + id2 * dim;
                if (dist_type == TYPE::DIST_COS)
                {
                    result += compare_cosDist(a, b, dim);
                }
                else if (dist_type == TYPE::DIST_COS_SIMILARITY)
                {
                    result += compare_cosSim(a, b, dim);
                }
                else if (dist_type == TYPE::DIST_EUCLIDEAN)
                {
                    float dist = 0.0;
                    float diff0, diff1, diff2, diff3;
                    const float *last = a + dim;
                    const float *unroll_group = last - 3;

                    /* Process 4 items with each loop for efficiency. */
                    while (a < unroll_group)
                    {
                        diff0 = a[0] - b[0];
                        diff1 = a[1] - b[1];
                        diff2 = a[2] - b[2];
                        diff3 = a[3] - b[3];
                        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                        a += 4;
                        b += 4;
                    }
                    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
                    while (a < last)
                    {
                        diff0 = *a++ - *b++;
                        dist += diff0 * diff0;
                    }

                    dist = std::sqrt(dist); // 开关 of weighted square
                    result += dist;
                }
                else
                {
                    std::cout << "Unknown dist_type!" << std::endl;
                    exit(-1);
                }
            }
            return result;
        }

        /**
         * compare_multi_weight():
         *      计算(base1中的)id1 和 (base2中的)id2 在field-group “group” 下的距离
         * @param base1 base1的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id1 base1 下的 ID
         * @param base2 base2的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id2 base2 下的 ID
         * @param dim_list
         * @param need_calcu field-group中涉及到需要进行距离计算的field编号
         * @param weight 每个field对应的distance计算权重，assert(weight.size() == smart_index->getFieldNum());
         * @param dist_type 距离计算方式
         */
        float compare_multi_weight(const std::vector<float *> &base1, unsigned id1,
                                   const std::vector<float *> &base2, unsigned id2,
                                   const std::vector<unsigned> &dim_list, const std::vector<int> &need_calcu,
                                   std::vector<float> &weight,
                                   TYPE dist_type = TYPE::DIST_EUCLIDEAN) const
        {
            float result = 0.0;
            for (auto field : need_calcu)
            {
                unsigned dim = dim_list[field];
                const float *a = base1[field] + id1 * dim;
                const float *b = base2[field] + id2 * dim;
                if (dist_type == TYPE::DIST_COS)
                {
                    result += weight[field] * compare_cosDist(a, b, dim);
                }
                else if (dist_type == TYPE::DIST_COS_SIMILARITY)
                {
                    result += weight[field] * compare_cosSim(a, b, dim);
                }
                else if (dist_type == TYPE::DIST_EUCLIDEAN)
                {
                    float dist = 0.0;
                    float diff0, diff1, diff2, diff3;
                    const float *last = a + dim;
                    const float *unroll_group = last - 3;

                    /* Process 4 items with each loop for efficiency. */
                    while (a < unroll_group)
                    {
                        diff0 = a[0] - b[0];
                        diff1 = a[1] - b[1];
                        diff2 = a[2] - b[2];
                        diff3 = a[3] - b[3];
                        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                        a += 4;
                        b += 4;
                    }
                    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
                    while (a < last)
                    {
                        diff0 = a[0] - b[0];
                        a++, b++;
                        dist += diff0 * diff0;
                    }

                    dist = std::sqrt(dist); // 开关 of weighted square
                    result += weight[field] * dist;
                }
                else
                {
                    std::cout << "Unknown dist_type!" << std::endl;
                    exit(-1);
                }
            }

            return static_cast<float>(result);
        }

        /**
         * compare_multi_weight():
         *      计算(base1中的)id1 和 (base2中的)id2 在field-group “group” 下的距离
         * @param base1 base1的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id1 base1 下的 ID
         * @param base2 base2的所有field的basedata，每个field中的basedata用一个float*的形式连续存储；
         * @param id2 base2 下的 ID
         * @param dim_list
         * @param need_calcu field-group中涉及到需要进行距离计算的field编号
         * @param weight 每个field对应的distance计算权重，assert(weight.size() == smart_index->getFieldNum());
         * @param max_dist 如果超过这个dist，则return = -1， 无需继续计算
         * @param dist_type 距离计算方式
         */
        float compare_multi_weight(const std::vector<float *> &base1, unsigned id1,
                                   const std::vector<float *> &base2, unsigned id2,
                                   const std::vector<unsigned> &dim_list, const std::vector<int> &need_calcu,
                                   std::vector<float> &weight, float max_dist,
                                   TYPE dist_type = TYPE::DIST_EUCLIDEAN) const
        {
            float result = 0.0;
            for (auto field : need_calcu)
            {
                unsigned dim = dim_list[field];
                const float *a = base1[field] + id1 * dim;
                const float *b = base2[field] + id2 * dim;
                if (dist_type == TYPE::DIST_COS)
                {
                    result += weight[field] * compare_cosDist(a, b, dim);
                    if (static_cast<float>(result) > max_dist)
                    {
                        return -1;
                    }
                }
                else if (dist_type == TYPE::DIST_COS_SIMILARITY)
                {
                    result += weight[field] * compare_cosSim(a, b, dim);
                    if (static_cast<float>(result) > max_dist)
                    {
                        return -1;
                    }
                }
                else if (dist_type == TYPE::DIST_EUCLIDEAN)
                {
                    // float dist = 0.0;
                    // float diff0, diff1, diff2, diff3;
                    float dist = 0.0;
                    float diff0, diff1, diff2, diff3;
                    const float *last = a + dim;
                    const float *unroll_group = last - 3;

                    /* Process 4 items with each loop for efficiency. */
                    while (a < unroll_group)
                    {
                        diff0 = a[0] - b[0];
                        diff1 = a[1] - b[1];
                        diff2 = a[2] - b[2];
                        diff3 = a[3] - b[3];
                        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                        a += 4;
                        b += 4;
                    }
                    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
                    while (a < last)
                    {
                        diff0 = *a++ - *b++;
                        dist += diff0 * diff0;
                    }
                    dist = std::sqrt(dist); // 开关 of weighted square
                    result += weight[field] * dist;
                    if (static_cast<float>(result) > max_dist)
                    {
                        return -1;
                    }
                }
                else
                {
                    exit(-1);
                }
            }
            return static_cast<float>(result);
        }
    };
}

template <typename T>
T compare(const T *a, const T *b, unsigned length)
{
    T dist = 0;

    float diff0, diff1, diff2, diff3;
    const T *last = a + length;
    const T *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group)
    {
        diff0 = a[0] - b[0];
        diff1 = a[1] - b[1];
        diff2 = a[2] - b[2];
        diff3 = a[3] - b[3];
        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        a += 4;
        b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last)
    {
        diff0 = *a++ - *b++;
        dist += diff0 * diff0;
    }

    dist = std::sqrt(dist); // 开关 of weighted square
    return dist;
}


template <typename T>
T compare_cosSim(const T *a, const T *b, unsigned length)
{
    T result = 0, ab = 0, aa = 0, bb = 0;

    float ab0, ab1, ab2, ab3, aa0, aa1, aa2, aa3, bb0, bb1, bb2, bb3;
    const T *last = a + length;
    const T *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group)
    {
        ab0 = a[0] * b[0];
        ab1 = a[1] * b[1];
        ab2 = a[2] * b[2];
        ab3 = a[3] * b[3];
        aa0 = a[0] * a[0];
        aa1 = a[1] * a[1];
        aa2 = a[2] * a[2];
        aa3 = a[3] * a[3];
        bb0 = b[0] * b[0];
        bb1 = b[1] * b[1];
        bb2 = b[2] * b[2];
        bb3 = b[3] * b[3];
        ab += ab0 + ab1 + ab2 + ab3;
        aa += aa0 + aa1 + aa2 + aa3;
        bb += bb0 + bb1 + bb2 + bb3;
        a += 4;
        b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last)
    {
        aa0 = a[0] * a[0];
        bb0 = b[0] * b[0];
        ab0 = *a++ * *b++;
        aa += aa0;
        ab += ab0;
        bb += bb0;
    }
    result = ab / (sqrt(aa) * sqrt(bb));

    return result;
}

template <typename T>
T compare_cosDist(const T *a, const T *b, unsigned length)
{
    T result = 0;
    result = 1 - compare_cosSim(a, b, length);

    return result;
}

#endif // XMT_DISTANCE_H
