#include "component.h"


// ======================================
// Neighborhood Construction
// ======================================
namespace xmt {

    // ----------------------------------------------------------------------------------------------------
    // PolyGraph
    // ----------------------------------------------------------------------------------------------------
    void ComponentRefineSmart::SetConfigs_multi(int group, float alpha)
    {
        smart_index->R = smart_index->getRRefineList()[group];
        smart_index->L = smart_index->getLRefineList()[group];
        smart_index->R_refine = smart_index->getRRefineList()[group];
        smart_index->ep_ = smart_index->each_ep_[group];
        smart_index->alpha = alpha;
        std::cout << "smart_index->L: " << smart_index->L << std::endl;
        std::cout << "smart_index->R: " << smart_index->R << std::endl;
        std::cout << "smart_index->R_refine: " << smart_index->R_refine << std::endl;
        std::cout << "smart_index->ep_: " << smart_index->ep_ << std::endl;
        std::cout << "smart_index->alpha: " << smart_index->alpha << std::endl;
    }

    void ComponentRefineSmart::RefineInner_multi(TYPE dist_type)
    {
        // std::mt19937 rng(rand());
        std::mt19937 rng(666);
        float alpha = 1.0;

        std::cout << "\n=========================================================" << std::endl;
        if (smart_index->getRefineRound() == 0)
        {
            // ENTRY
            auto *a = new ComponentRefineEntryCentroid_multi(smart_index);
            a->EntryInner_multi();
        }

        // REFINE
        smart_index->addRefineRound();
        std::cout << "__Start REFINT ROUND" << smart_index->getRefineRound() << "__" << std::endl;
        for (int group = smart_index->getGroupNum() - 1; group >= 0; group--)
        {
            std::cout << "__START REFINE-" << smart_index->getRefineRound() << ": for group " << group << "__" << std::endl;
            std::cout << "  with group element: ";
            for (auto f : smart_index->getGroupList()[group])
            {
                std::cout << f << ", ";
            }
            std::cout << std::endl;
            RefineInner_multi_4group(group, alpha, true, dist_type);
            std::cout << "__END REFINE-" << smart_index->getRefineRound() << ": for group " << group << "__\n\n"
                      << std::endl;
        }
        std::cout << "\n=========================================================" << std::endl;

        smart_index->setAlphaNow(smart_index->getAlpha2());
    }

    void ComponentRefineSmart::RefineInner_multi_4group(int group, float alpha, bool hint, TYPE dist_type)
    {
        auto s = std::chrono::high_resolution_clock::now();

        SetConfigs_multi(group, alpha);

        smart_index->getFinalGraph(group).resize(smart_index->getBaseLen());

        // ### 1. 拷贝一版旧图
        smart_index->copyOldGraphList();

        // ### 2. 根据旧图进行refine，直接对FinalGraph进行修改，作为新图
        Link_multi_4group(group, dist_type); // 在这部分直接对FinalGraph进行修改

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ 【 Refine for group " << group << " 】: time is " << load_info_time.count() << " ^^^^^^ \n"
                  << std::endl;
        if (hint)
        {
            // ## Show InitGraph Summary
            std::vector<int> groupList = {group};
            smart_index->summaryFinalGraph(groupList, {0, 1, smart_index->getBaseLen() - 1});
        }

        // ### 3. 删除旧图副本，释放空间
        smart_index->clearOldGraphList();
    }

    void ComponentRefineSmart::Link_multi_4group(int group, TYPE dist_type)
    {
        std::vector<std::mutex> locks(smart_index->getBaseLen());

        // CANDIDATE
        std::cout << "__CANDIDATE : AGS__" << std::endl;
        ComponentCandidate_multi *a = new ComponentCandidateAGS_multi(smart_index);

        // PRUNE
        std::cout << "__PRUNE : PolyGraph__" << std::endl;
        ComponentPrune_multi *b = new ComponentPrunePolyGraph_multi(smart_index);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> candidate_time(0.0), prune_time(0.0), inter_time(0.0), inter_insert_time(0.0), inter_prune_time(0.0);

        // ## show relaGroups
        std::cout << "__needUseRelaIndex : ";
        for (auto ind : smart_index->getRelaCheck(group))
        {
            std::cout << ind << ", ";
        }
        std::cout << "___" << std::endl;
        unsigned count_num = 0;

#pragma omp parallel
        {
            std::vector<MultiIndex::SimpleNeighbor> pool;
            std::vector<MultiIndex::SimpleNeighbor> pool_cand;
            pool.reserve(smart_index->getBaseLen());
            pool_cand.reserve(smart_index->getBaseLen());
            boost::dynamic_bitset<> flags(smart_index->getBaseLen(), 0);
            std::chrono::duration<double> cand_time_local(0.0), prune_time_local(0.0);

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                pool.clear();
                pool_cand.clear(); 
                flags.reset();
                a->CandidateInner_multi_4group_STAR_allIndex(n, smart_index->ep_, group, flags, pool, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                cand_time_local += e - s;

                s = std::chrono::high_resolution_clock::now();
                b->PruneInner_multi_4group_withOthers(n, group, pool, locks, dist_type);
                e = std::chrono::high_resolution_clock::now();
                prune_time_local += e - s;
                
                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(cand+prune)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                candidate_time += cand_time_local;
                prune_time += prune_time_local;
            }
        }

        count_num = 0;
#pragma omp parallel
        {
            std::chrono::duration<double> inter_insert_time_local(0.0);
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                InterInsert_multi_4group_insert(n, group, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                inter_insert_time_local += e - s;
                
                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(InterInsert)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                inter_insert_time += inter_insert_time_local;
                inter_time += inter_insert_time_local;
            }
        }
       
        std::cout << "######################" << std::endl;
        std::cout << "# candidate_time: " << candidate_time.count() << std::endl;
        std::cout << "# prune_time: " << prune_time.count() << std::endl;
        std::cout << "# inter_time: " << inter_time.count() << std::endl;
        std::cout << "#     inter_insert_time: " << inter_insert_time.count() << std::endl;
        std::cout << "#     inter_prune_time: " << inter_prune_time.count() << std::endl;
        std::cout << "######################" << std::endl;

        smart_index->alpha = smart_index->getParam().get<float>("alpha2");        
    }

    void ComponentRefineSmart::InterInsert_multi_4group_insert(unsigned int n, int group, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        std::unique_lock<std::mutex> guard_src(locks[n]);
        const auto src_pool = smart_index->getFinalGraph(group)[n];
        guard_src.unlock();

        for (size_t i = 0; i < src_pool.size(); i++)
        {
            MultiIndex::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            {
                std::unique_lock<std::mutex> guard(locks[des]);
                auto &des_pool = smart_index->getFinalGraph(group)[des];
                bool dup = false;

                // ## (1) Directly Insert
                {
                    for (auto &x : des_pool)
                    {
                        if (x.id == n)
                        {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup)
                    {
                        des_pool.emplace_back(sn);
                    }
                }

                // ## (2) IF NEED PRUNE
                if (des_pool.size() > smart_index->R) // ## (2) INSERT & PRUNE
                {
                    auto temp_pool = smart_index->getFinalGraph(group)[des];
                    guard.unlock(); 
                    ComponentPrune_multi *b = new ComponentPrunePolyGraph_multi(smart_index);
                    b->PruneInner_multi_4group_withOthers(des, group, temp_pool, locks, dist_type);
                }
            }
        }
    }

    // void ComponentRefineSmart::InterInsert_multi_4group_prune(unsigned int n, int group, std::vector<std::mutex> &locks,
    //                                                           TYPE dist_type)
    // {
    //     std::unique_lock<std::mutex> guard(locks[n]);
    //     std::vector<xmt::MultiIndex::SimpleNeighbor> pool = smart_index->getOutNeigh(group, n);
    //     guard.unlock();
    //     // ## (2) 只有超出R才会进行prune
    //     if (pool.size() > smart_index->R)
    //     {
    //         ComponentPrune_multi *b = new ComponentPrunePolyGraph_multi(smart_index);
    //         b->PruneInner_multi_4group_withOthers(n, group, pool, locks, dist_type);
    //     }
    // }
    




    // ----------------------------------------------------------------------------------------------------
    // Oracle
    // ----------------------------------------------------------------------------------------------------
    void ComponentRefineSmart_Oracle::SetConfigs_multi(int group, float alpha)
    {
        smart_index->R = smart_index->getRRefineList()[group];
        smart_index->L = smart_index->getLRefineList()[group];
        smart_index->R_refine = smart_index->getRRefineList()[group];
        smart_index->ep_ = smart_index->each_ep_[group];
        smart_index->alpha = alpha;
        std::cout << "smart_index->L: " << smart_index->L << std::endl;
        std::cout << "smart_index->R: " << smart_index->R << std::endl;
        std::cout << "smart_index->R_refine: " << smart_index->R_refine << std::endl;
        std::cout << "smart_index->ep_: " << smart_index->ep_ << std::endl;
        std::cout << "smart_index->alpha: " << smart_index->alpha << std::endl;
    }

    void ComponentRefineSmart_Oracle::RefineInner_multi(TYPE dist_type)
    {
        // std::mt19937 rng(rand());
        std::mt19937 rng(666);
        float alpha = 1.0;

        std::cout << "\n=========================================================" << std::endl;
        if (smart_index->getRefineRound() == 0)
        {
            // ENTRY
            auto *a = new ComponentRefineEntryCentroid_multi(smart_index);
            a->EntryInner_multi();
        }
        else
        {
            alpha = smart_index->getAlphaNow();
        }

        // REFINE
        smart_index->addRefineRound();
        std::cout << "__Start REFINT ROUND" << smart_index->getRefineRound() << "__" << std::endl;
        for (int group = smart_index->getGroupNum() - 1; group >= 0; group--)
        {
            std::cout << "__START REFINE-" << smart_index->getRefineRound() << ": ORACLE for group " << group << "__" << std::endl;
            std::cout << "  with group element: ";
            for (auto f : smart_index->getGroupList()[group])
            {
                std::cout << f << ", ";
            }
            std::cout << std::endl;
            RefineInner_multi_4group(group, alpha, true, dist_type); // ComponentRefineSmart::
            std::cout << "__END REFINE-" << smart_index->getRefineRound() << ": ORACLE for group " << group << " with this-round-alpha = " << alpha << "__\n\n"
                      << std::endl;
        }
        std::cout << "\n=========================================================" << std::endl;

        smart_index->setAlphaNow(smart_index->getAlpha2());
    }

    void ComponentRefineSmart_Oracle::RefineInner_multi_4group(int group, float alpha, bool hint, TYPE dist_type)
    {
        auto s = std::chrono::high_resolution_clock::now();

        SetConfigs_multi(group, alpha);

        smart_index->getFinalGraph(group).resize(smart_index->getBaseLen());

        // ### 1. 拷贝一版旧图
        smart_index->copyOldGraphList();

        // ### 2. 根据旧图进行refine，直接对FinalGraph进行修改，作为新图
        Link_multi_4group(group, dist_type); // 在这部分直接对FinalGraph进行修改

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ 【 Refine for group " << group << " 】: time is " << load_info_time.count() << " ^^^^^^ \n"
                  << std::endl;
        if (hint)
        {
            // ## Show InitGraph Summary
            std::vector<int> groupList = {group};
            smart_index->summaryFinalGraph(groupList, {0, 1, smart_index->getBaseLen() - 1});
        }

        // ### 3. 删除旧图副本，释放空间
        smart_index->clearOldGraphList();
    }

    void ComponentRefineSmart_Oracle::Link_multi_4group(int group, TYPE dist_type)
    {
        std::vector<std::mutex> locks(smart_index->getBaseLen());

        std::cout << "alpha " << smart_index->alpha << std::endl;

        // CANDIDATE
        std::cout << "__CANDIDATE: OracleVamana__" << std::endl;
        ComponentCandidate_multi *a = new ComponentCandidateAGS_multi(smart_index);

        // PRUNE
        std::cout << "__PRUNE: OracleVamana__" << std::endl;
        ComponentPrune_multi *b = new ComponentPruneVamana_multi(smart_index);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> candidate_time(0.0), prune_time(0.0), inter_time(0.0), inter_insert_time(0.0), inter_prune_time(0.0);

        // ## show relaGroups
        std::cout << "__needUseRelaIndex : ";
        for (auto ind : smart_index->getRelaCheck(group))
        {
            std::cout << ind << ", ";
        }
        std::cout << "___" << std::endl;
        unsigned count_num = 0;

#pragma omp parallel
        {
            std::vector<MultiIndex::SimpleNeighbor> pool;
            std::vector<MultiIndex::SimpleNeighbor> pool_cand;
            pool.reserve(smart_index->getBaseLen());
            pool_cand.reserve(smart_index->getBaseLen());
            boost::dynamic_bitset<> flags(smart_index->getBaseLen(), 0);
            std::chrono::duration<double> cand_time_local(0.0), prune_time_local(0.0);

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                pool.clear();
                pool_cand.clear(); 
                flags.reset();
                a->CandidateInner_multi_4group_STAR(n, smart_index->ep_, group, flags, pool, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                cand_time_local += e - s;
                
                s = std::chrono::high_resolution_clock::now();
                b->PruneInner_multi_4group_withOthers(n, group, pool, locks, dist_type);
                e = std::chrono::high_resolution_clock::now();
                prune_time_local += e - s;

                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(cand+prune)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                candidate_time += cand_time_local;
                prune_time += prune_time_local;
            }
        }

        count_num = 0;
#pragma omp parallel
        {
            std::chrono::duration<double> inter_insert_time_local(0.0);
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                InterInsert_multi_4group_insert(n, group, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                inter_insert_time_local += e - s;

                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(InterInsert)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                inter_insert_time += inter_insert_time_local;
                inter_time += inter_insert_time_local;
            }
        }

        std::cout << "######################" << std::endl;
        std::cout << "# candidate_time: " << candidate_time.count() << std::endl;
        std::cout << "# prune_time: " << prune_time.count() << std::endl;
        std::cout << "# inter_time: " << inter_time.count() << std::endl;
        std::cout << "#     inter_insert_time: " << inter_insert_time.count() << std::endl;
        std::cout << "#     inter_prune_time: " << inter_prune_time.count() << std::endl;
        std::cout << "######################" << std::endl;

        smart_index->alpha = smart_index->getParam().get<float>("alpha2");
    }

    void ComponentRefineSmart_Oracle::InterInsert_multi_4group_insert(unsigned int n, int group, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        std::unique_lock<std::mutex> guard_src(locks[n]);
        const auto src_pool = smart_index->getFinalGraph(group)[n];
        guard_src.unlock();

        for (size_t i = 0; i < src_pool.size(); i++)
        {
            MultiIndex::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            {
                std::unique_lock<std::mutex> guard(locks[des]);
                auto &des_pool = smart_index->getFinalGraph(group)[des];
                bool dup = false;

                // ## (1) Directly Insert
                {
                    for (auto &x : des_pool)
                    {
                        if (x.id == n)
                        {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup)
                    {
                        des_pool.emplace_back(sn);
                    }
                }

                // ## (2) IF NEED PRUNE
                if (des_pool.size() > smart_index->R) // ## (2) INSERT & PRUNE
                {
                    auto temp_pool = smart_index->getFinalGraph(group)[des];
                    guard.unlock(); 
                    // ComponentPrune_multi *b = new ComponentPrunePolyGraph_multi(smart_index);
                    ComponentPrune_multi *b = new ComponentPruneVamana_multi(smart_index);
                    b->PruneInner_multi_4group_withOthers(des, group, temp_pool, locks, dist_type);
                }
            }
        }
    }


    // ----------------------------------------------------------------------------------------------------
    // Vamana-series
    // ----------------------------------------------------------------------------------------------------
    void ComponentRefineVamana_multi::SetConfigs_multi(int group, float alpha)
    {
        smart_index->R = smart_index->getRRefineList()[group];
        smart_index->L = smart_index->getLRefineList()[group];
        smart_index->R_refine = smart_index->getRRefineList()[group];
        smart_index->ep_ = smart_index->each_ep_[group];
        smart_index->alpha = alpha;
        std::cout << "smart_index->L: " << smart_index->L << std::endl;
        std::cout << "smart_index->R: " << smart_index->R << std::endl;
        std::cout << "smart_index->R_refine: " << smart_index->R_refine << std::endl;
        std::cout << "smart_index->ep_: " << smart_index->ep_ << std::endl;
        std::cout << "smart_index->alpha: " << smart_index->alpha << std::endl;
    }

    void ComponentRefineVamana_multi::RefineInner_multi(TYPE dist_type)
    {
        // std::mt19937 rng(rand());
        std::mt19937 rng(666);
        float alpha = 1.0;

        std::cout << "\n=========================================================" << std::endl;
        if (smart_index->getRefineRound() == 0)
        {
            // ENTRY
            auto *a = new ComponentRefineEntryCentroid_multi(smart_index);
            a->EntryInner_multi();
        }
        else
        {
            alpha = smart_index->getAlphaNow();
        }

        // REFINE
        smart_index->addRefineRound();
        std::cout << "__Start REFINT ROUND" << smart_index->getRefineRound() << "__" << std::endl;
        for (int group = smart_index->getGroupNum() - 1; group >= 0; group--)
        {
            std::cout << "__START REFINE-" << smart_index->getRefineRound() << ": Vamana for group " << group << "__" << std::endl;
            std::cout << "  with group element: ";
            for (auto f : smart_index->getGroupList()[group])
            {
                std::cout << f << ", ";
            }
            std::cout << std::endl;
            RefineInner_multi_4group(group, alpha, true, dist_type);
            std::cout << "__END REFINE-" << smart_index->getRefineRound() << ": Vamana for group " << group << " with this-round-alpha = " << alpha << "__\n\n"
                      << std::endl;
        }
        std::cout << "\n=========================================================" << std::endl;

        smart_index->setAlphaNow(smart_index->getAlpha2());
    }

    void ComponentRefineVamana_multi::RefineInner_multi_4group(int group, float alpha, bool hint, TYPE dist_type)
    {
        auto s = std::chrono::high_resolution_clock::now();

        SetConfigs_multi(group, alpha);

        smart_index->getFinalGraph(group).resize(smart_index->getBaseLen());

        // ### 1. 拷贝一版旧图
        smart_index->copyOldGraphList();

        // ### 2. 根据旧图进行refine，直接对FinalGraph进行修改，作为新图
        Link_multi_4group(group, dist_type); // 在这部分直接对FinalGraph进行修改

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_info_time = e - s;
        std::cout << "^^^^^^ 【 Refine for group " << group << " 】: time is " << load_info_time.count() << " ^^^^^^ \n" << std::endl;
        if (hint)
        {
            // ## Show InitGraph Summary
            std::vector<int> groupList = {group};
            smart_index->summaryFinalGraph(groupList, {0, 1, smart_index->getBaseLen() - 1});
        }

        // ### 3. 删除旧图副本，释放空间
        smart_index->clearOldGraphList();
    }

    void ComponentRefineVamana_multi::Link_multi_4group(int group, TYPE dist_type)
    {
        std::vector<std::mutex> locks(smart_index->getBaseLen());

        std::cout << "alpha " << smart_index->alpha << std::endl;

        // CANDIDATE
        std::cout << "__CANDIDATE: Vamana__" << std::endl;
        ComponentCandidate_multi *a = new ComponentCandidateAGS_multi(smart_index);

        // PRUNE
        std::cout << "__PRUNE : Vamana__" << std::endl;
        ComponentPrune_multi *b = new ComponentPruneVamana_multi(smart_index);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> candidate_time(0.0), prune_time(0.0), inter_time(0.0), inter_insert_time(0.0), inter_prune_time(0.0);

        // ## show relaGroups
        std::cout << "__needUseRelaIndex : ";
        for (auto ind : smart_index->getRelaCheck(group))
        {
            std::cout << ind << ", ";
        }
        std::cout << "___" << std::endl;
        unsigned count_num = 0;

#pragma omp parallel
        {
            std::vector<MultiIndex::SimpleNeighbor> pool;
            std::vector<MultiIndex::SimpleNeighbor> pool_cand;
            pool.reserve(smart_index->getBaseLen());
            pool_cand.reserve(smart_index->getBaseLen());
            boost::dynamic_bitset<> flags(smart_index->getBaseLen(), 0);
            std::chrono::duration<double> cand_time_local(0.0), prune_time_local(0.0);

#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                pool.clear();
                pool_cand.clear(); 
                flags.reset();
                a->CandidateInner_multi_4group_STAR_allIndex(n, smart_index->ep_, group, flags, pool, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                cand_time_local += e - s;
                
                s = std::chrono::high_resolution_clock::now();
                b->PruneInner_multi_4group_withOthers(n, group, pool, locks, dist_type);
                e = std::chrono::high_resolution_clock::now();
                prune_time_local += e - s;

                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(cand+prune)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                candidate_time += cand_time_local;
                prune_time += prune_time_local;
            }
        }

        count_num = 0;
#pragma omp parallel
        {
            std::chrono::duration<double> inter_insert_time_local(0.0);
#pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < smart_index->getBaseLen(); ++n)
            {
                auto s = std::chrono::high_resolution_clock::now();
                InterInsert_multi_4group_insert(n, group, locks, dist_type);
                auto e = std::chrono::high_resolution_clock::now();
                inter_insert_time_local += e - s;

                count_num++;
                if (count_num % 100000 == 0)
                {
                    auto now = std::chrono::system_clock::now();
                    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                    std::tm local_tm = *std::localtime(&now_time);
                    std::cout << "-- Build process(Link_multi_4group(InterInsert)): " << count_num << " / " << smart_index->getBaseLen()
                              << " at time: " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
                }
            }
#pragma omp critical
            {
                inter_insert_time += inter_insert_time_local;
                inter_time += inter_insert_time_local;
            }
        }

        std::cout << "######################" << std::endl;
        std::cout << "# candidate_time: " << candidate_time.count() << std::endl;
        std::cout << "# prune_time: " << prune_time.count() << std::endl;
        std::cout << "# inter_time: " << inter_time.count() << std::endl;
        std::cout << "#     inter_insert_time: " << inter_insert_time.count() << std::endl;
        std::cout << "#     inter_prune_time: " << inter_prune_time.count() << std::endl;
        std::cout << "######################" << std::endl;

        smart_index->alpha = smart_index->getParam().get<float>("alpha2");
    }

    void ComponentRefineVamana_multi::InterInsert_multi_4group_insert(unsigned int n, int group, std::vector<std::mutex> &locks, TYPE dist_type)
    {
        std::unique_lock<std::mutex> guard_src(locks[n]);
        const auto src_pool = smart_index->getFinalGraph(group)[n];
        guard_src.unlock();

        for (size_t i = 0; i < src_pool.size(); i++)
        {
            MultiIndex::SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            {
                std::unique_lock<std::mutex> guard(locks[des]);
                auto &des_pool = smart_index->getFinalGraph(group)[des];
                bool dup = false;

                // ## (1) Directly Insert
                {
                    for (auto &x : des_pool)
                    {
                        if (x.id == n)
                        {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup)
                    {
                        des_pool.emplace_back(sn);
                    }
                }

                // ## (2) IF NEED PRUNE
                if (des_pool.size() > smart_index->R) // ## (2) INSERT & PRUNE
                {
                    auto temp_pool = smart_index->getFinalGraph(group)[des];
                    guard.unlock(); 
                    ComponentPrune_multi *b = new ComponentPruneVamana_multi(smart_index);
                    b->PruneInner_multi_4group_withOthers(des, group, temp_pool, locks, dist_type);
                }
            }
        }
    }
}