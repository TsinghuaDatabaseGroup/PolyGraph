#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <regex>
#include <cassert>

#include "index_alg_fun.h"
#include "python_file/run_python.h"
#include "exp_data.h"



// ============ /*  Commands that can be used:  */ =====================================================
// // -- [Different "DATASET" name] --
// ImageText
// QA2
// Wiki
// Protein

// // -- [ "INDEX_NAME" of Baselines/Oracle ] --
// hnsw_fusion
// vamana_fusion
// vamana_equNoTotal
// vamana_allWeight
// vamana_oracle

// // -- [PolyGraph] --
// ./main PolyGraph [DATASET] build -rela_use_intersect 0 -total_sim_thresh 0.95 -rela_sim_thresh 0.5
// ./main PolyGraph [DATASET] all_recall_search 20 -search_rela_sim_thresh 0.5 --> FallbackIntersect
// ./main PolyGraph [DATASET] all_recall_search_allIndex 20
// ./main PolyGraph [DATASET] all_recall_search_intersect 20

// // -- [Baselines] --
// ./main [INDEX_NAME] [DATASET] build
// ./main [INDEX_NAME] [DATASET] all_recall_search [k]

// 可添加其他调节超参数的输入
// -L R
// -L_refine L
// -R_refine R
// ============ /*  Commands that can be used, END */ =====================================================

// // ⚠️： 需要创建python的虚拟环境
// 1. 创建虚拟环境
// cd /home/mengtong/MyWork/PolyGraph
// python3 -m venv ./pythonEnv_ForSI
// source ./pythonEnv_ForSI/bin/activate
// python -m pip install --upgrade pip setuptools wheel

// 2. 安装你的脚本需要的包
// 2.1 建议保存 requirements.txt
// 2.1.1 为了以后重建环境方便，可以创建：
// cat > /home/mengtong/MyWork/PolyGraph/include/python_file/requirements.txt << 'EOF'
// numpy
// scipy
// scikit-learn
// matplotlib
// joblib
// tqdm
// EOF
// 2.2 根据requirements.txt安装脚本需要的包
// 2.2.1 以后如果环境坏了，直接：
// source /home/mengtong/MyWork/PolyGraph/pythonEnv/bin/activate
// pip install -r /home/mengtong/MyWork/PolyGraph/include/python_file/requirements.txt
// 2.2.2 验证是否安装成功：
// ./pythonEnv_ForSI/bin/python3 -c "import numpy, scipy, sklearn, matplotlib, joblib, tqdm; print('Python env OK')"


// // -- 在miniconda中安装boost --
// // 1. 首先打开我的conda环境：base
// conda activate base
// // 2. 向conda中安装boost：
// source /home/mengtong/miniconda3/etc/profile.d/conda.sh
// conda activate base
// conda install -c conda-forge boost-cpp
// // 3. 检查是否安装成功
// find $CONDA_PREFIX -name dynamic_bitset.hpp 2>/dev/null | head


// float ObjectOverall::singleDistUB = floatMax; // 初始化静态成员变量
std::regex paraName("-(\\w+)");

int main(int totalArgc, char **argv)
{
    unsigned K = 20;
    int argc = 0;

    for (int i = 0; i < totalArgc; i++)
    {
        if (std::regex_match(argv[i], paraName))
        {
            break;
        }
        argc++;
    }
    std::cout << "totalArgc: " << totalArgc << ", argc: " << argc << std::endl;

    std::string alg(argv[1]);
    std::string dataset(argv[2]);
    std::string exc_type(argv[3]);
    std::string dist_type = "euclidean";
    std::string dataset_root = R"(../dataset/)";
    std::string index_path = R"(../myIndex)";
    std::string graph_file("../myIndex/" + alg + "_" + dataset + ".graph");


    xmt::Parameters parameters;
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<std::string>("index_path", index_path);
    // XMT修改：应为spark10有40个CPU kernels，把n_threads提高到30
    // parameters.set<unsigned>("n_threads", 8);
    parameters.set<unsigned>("n_threads", 30);
    parameters.set<std::string>("dist_type", dist_type);
    parameters.set<std::string>("graph_file", graph_file);
    parameters.set<std::string>("exc_type", exc_type);
    parameters.set<std::string>("dataset", dataset);
    parameters.set<std::string>("alg", alg);

    std::cout << "exc_type: " << exc_type << std::endl;
    std::cout << "algorithm: " << alg << std::endl;
    std::cout << "dataset: " << dataset << std::endl;


    // ======= 1. prepare datafile path and hyperparameter values =============================================
    // For search
    if (argc == 5) 
    {
        if (exc_type == "search" || exc_type == "ground_truth"
            || exc_type == "all_recall_search" || exc_type == "all_recall_search_intersect" || exc_type == "all_recall_search_allIndex"
            || exc_type == "recall_search" || exc_type == "recall_search_intersect" || exc_type == "recall_search_allIndex" 
            || exc_type == "get_ground_truth" || exc_type == "get_ground_truth_one_to_one"
        )
        {
            K = (unsigned)atoi(argv[4]);
            std::cout << "search with numTop = " << K << std::endl;
            parameters.set<unsigned>("K_search", K);
            parameters.set<unsigned>("L_search", K);
        }
        else
        {
            std::cout << "./main PolyGraph dataset search [K_search]" << std::endl;
            exit(-1);
        }
    }

    // check exc_type parameters: setting parameters as default
    if (exc_type == "build" 
        || exc_type == "search" || exc_type == "ground_truth"
        || exc_type == "all_recall_search" || exc_type == "all_recall_search_intersect" || exc_type == "all_recall_search_allIndex"
        || exc_type == "recall_search" || exc_type == "recall_search_intersect" || exc_type == "recall_search_allIndex"
        || exc_type == "get_ground_truth" || exc_type == "get_ground_truth_one_to_one"
    )
    {
        set_para(alg, dataset, parameters);
        std::cout << "__ exc_type is: " << exc_type << " __" << std::endl;
    }
    else
    {
        std::cout << "Maybe exc_type is wrong: " << exc_type << std::endl;
        exit(-1);
    }

    // --- change hyperparameters according to cmd -------
    // | 对于 build，判断是否有传入&定义 -alpha2, -L, -L_refine, -R ，等超参数；如果有，则根据传入数值进行hyper parameter修改。
    // -------------------------------------------------
    float total_sim_thresh = 0.95, rela_sim_thresh = 0.5, search_rela_sim_thresh = 0.5;
    unsigned max_group = 0;
    bool rela_use_intersect = true;
    for (int i = argc; i < totalArgc; i++)
    {
        std::smatch paraMatcher;
        std::string paraType(argv[i]);

        if (std::regex_match(paraType, paraMatcher, paraName))
        {
            if (i + 1 >= totalArgc)
            {
                std::cerr << "Missing value for parameter: " << paraType << std::endl;
                break;
            }
            i++; // move to value
            std::string valueStr(argv[i]);

            // Try to parse as unsigned int
            std::stringstream ss(valueStr);
            unsigned uval;
            ss >> uval;

            if (!ss.fail() && ss.eof() 
                && 
                ( paraMatcher.str(1) != "total_sim_thresh" || paraMatcher.str(1) != "rela_sim_thresh"
                 || paraMatcher.str(1) != "search_rela_sim_thresh" ))
            {
                if (paraMatcher.str(1) == "alpha2" ) {
                    std::cerr << "Not allow to change alpha2" << std::endl;
                    exit(-1);
                }
                else if (paraMatcher.str(1) == "rela_use_intersect") {
                    if (uval == 0) 
                    { 
                        rela_use_intersect = false; 
                        std::cout << "--rela_use_intersect = " << false << std::endl;
                    }
                    parameters.set<unsigned>(paraMatcher.str(1), uval);
                    std::cout << "set:     " << paraMatcher.str(1) << "(unsigned): " << uval << std::endl;
                }
                else if (paraMatcher.str(1) == "max_group") {
                    max_group = uval;
                    parameters.set<unsigned>(paraMatcher.str(1), uval);
                    std::cout << "set:     " << paraMatcher.str(1) << "(unsigned): " << uval << std::endl;
                }
                else {
                    parameters.set<unsigned>(paraMatcher.str(1), uval);
                    std::cout << "set:     " << paraMatcher.str(1) << "(unsigned): " << uval << std::endl;
                }
            }
            else
            {
                float fval = std::stof(valueStr);
                parameters.set<float>(paraMatcher.str(1), fval);
                std::cout << "set:     " << paraMatcher.str(1) << "(float): " << fval << std::endl;
                if (paraMatcher.str(1) == "total_sim_thresh") {total_sim_thresh = fval;}
                else if (paraMatcher.str(1) == "rela_sim_thresh") {rela_sim_thresh = fval;}
                else if (paraMatcher.str(1) == "search_rela_sim_thresh") {search_rela_sim_thresh = fval;}
            }
        }
    }
    if (!parameters.exist("search_rela_sim_thresh")) {
        parameters.set<float>("search_rela_sim_thresh", 0.5);
    }


    // ====== 2. run_python(): Do Representative Weight Selection =================================
    // if (exc_type == "cluster_build") {
    if (alg == "PolyGraph" && exc_type == "build") {
        bool override = true;

        // -- python路径
        std::string python_path = "../include/python_file/cluster_Now_distSimThresh_NoOrderRedecide_withCacheAligned_v2.py"; 
        std::vector<std::string> extra_args;
        extra_args.emplace_back("--search_corr_thresh");
        extra_args.emplace_back(std::to_string(search_rela_sim_thresh));

        // 输出结果在cluster_groups.txt中；本身自己有backup
        run_python(
            override, rela_use_intersect, total_sim_thresh, rela_sim_thresh, max_group,
            python_path,
            "../include/python_file/nohup_logs/bash.log", // 这次python的运行过程信息存储路径的前缀；后会加详细时间（应该是比c++中展示的时间略小且约等于）
            "../pythonEnv_ForSI/bin/python3", //venv_python
            parameters.get<std::string>("txt_path"), // path_info.txt
            parameters.get<std::string>("weight_path"), // useWeightEachQuery.txt
            extra_args
        );

        // 获取当前时间点
        auto now = std::chrono::system_clock::now();
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm* now_tm = std::localtime(&now_time_t);
        std::cout << "-- Record time after RUN PYTHON is: " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    }




    // ===== 3. build or search ============================================
    if (alg == "hnsw_fusion")
    {
        HNSW_FUSION(parameters);
    }
    else if (alg == "PolyGraph")
    {
        PolyGraph(parameters);
    }
    else if (alg == "vamana_equNoTotal" || alg == "vamana_fusion" || alg == "vamana_allWeight" || alg == "vamana_oracle")
    {
        vamana_baseline_index_group(parameters);
    }
    else
    {
        std::cout << "alg input error!\n";
        exit(-1);
    }

    std::cout << "Here End!" << std::endl;
    return 0;
}