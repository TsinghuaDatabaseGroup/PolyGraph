#include "component.h"


// ======================================
// load data
//      For: load()
// ======================================
namespace xmt {
    inline void load_data_txt(char *filename, float *&data) {
        std::ifstream in(filename, std::ios::in);
        if (!in.is_open()) {
            std::cerr << "open file error-12" << std::endl;
            exit(-1);
        }
        int n = 0;
        data = new float[150 * 2];

        char buffer[256];
        while(!in.eof()) {
            in.getline(buffer, 50);

            int i = 0;
            while(buffer[i] < '0' || buffer[i] > '9') i ++;

            std::string s = "";
            while(buffer[i] >= '0' && buffer[i] <= '9') {
                s += buffer[i];
                i ++;
            }
            float num = std::stof(s);
            data[n] = num; n ++;

            while(buffer[i] < '0' || buffer[i] > '9') i ++;

            s = "";
            while(buffer[i] >= '0' && buffer[i] <= '9') {
                s += buffer[i];
                i ++;
            }
            float num2 = std::stof(s);
            data[n] = num2; n ++;
        }
        in.close();
    }


    /**
     * 1. load parameters and various datasets (according to path in file "")
     * @param ground_file *_groundtruth.ivecs
     * @param parameters
     */
    void ComponentLoad_multi::LoadInner_multi(Parameters &parameters)
    {
        // path for various datasets
        std::vector<std::string> base_path_list;
        std::vector<std::string> query_path_list;
        std::vector<std::string> graph_path_list;
        std::string txt_path = parameters.get<std::string>("txt_path");
        ComponentLoad_multi::set_path_from_txt(txt_path, base_path_list, query_path_list, true);
        // [base_path_list] check
        std::cout << "### [base_path_list] check ### " << std::endl;
        for (int i = 0; i < base_path_list.size(); i++)
        {
            std::cout << base_path_list[i] << std::endl;
        }
        std::cout << "### [base_path_list] check end ###" << std::endl;

        // ## 1. load parameters and various datasets (according to path in file "")
        // ## 1.1. base_data for each field
        unsigned n_assert(0);
        smart_index->setFieldNum(base_path_list.size());
        for (int f = 0; f < smart_index->getFieldNum(); f++)
        {
            float *data = nullptr;
            unsigned n{};
            unsigned dim{};
            load_data<float>(&base_path_list[f][0], data, n, dim);
            smart_index->emplaceBaseData(data);
            smart_index->setBaseLen(n);
            smart_index->emplaceBaseDim(dim);

            if (f)
            {
                assert(n == n_assert);
            }
            else
            {
                n_assert = n;
            }
            assert(smart_index->getBaseData(f) != nullptr && smart_index->getBaseLen() != 0 && smart_index->getBaseDim(f) != 0);
        }

        // ## 1.2. query_data for each field
        for (int f = 0; f < query_path_list.size(); f++)
        {
            float *query_data = nullptr;
            unsigned query_n{};
            unsigned query_dim{};
            load_data<float>(&query_path_list[f][0], query_data, query_n, query_dim);
            smart_index->emplaceQueryData(query_data);
            smart_index->setQueryLen(query_n);
            smart_index->emplaceQueryDim(query_dim);

            if (f)
            {
                assert(query_n == n_assert);
            }
            else
            {
                n_assert = query_n;
            }
            assert(smart_index->getQueryData(f) != nullptr && query_n > 0 && smart_index->getQueryDim(f) == smart_index->getBaseDim(f));
        }

        // ## 1.3. ground_data.
        if (parameters.get<std::string>("exc_type") != "build")
        {
            unsigned *ground_data = new unsigned[smart_index->getQueryLen() * parameters.get<unsigned>("K_search")];
            smart_index->setGroundData(ground_data);
            smart_index->setGroundLen(smart_index->getQueryLen());
            smart_index->setGroundDim(parameters.get<unsigned>("K_search"));
        }

        smart_index->setParam(parameters);
    }


    /**
     * ComponentLoad_multi::set_path_from_txt():
     *      根据“txt_path”对应的txt文件分配base_path_list, query_path_list, graph_path_list
     *      txt_path格式为：
     *              base_path
     *              query_path
     * @param txt_path *_.txt
     * @param base_path_list(&) list of all base-data path
     * @param query_path_list(&) list of all query-data path
     */
    void ComponentLoad_multi::set_path_from_txt(std::string txt_path,
                                                std::vector<std::string> &base_path_list,
                                                std::vector<std::string> &query_path_list,
                                                bool show_summary)
    {
        std::vector<std::string> graph_path_list;
        std::ifstream file(txt_path);
        std::string line;

        if (!file.is_open())
        {
            std::cout << "Unable to open file: " << txt_path << std::endl;
            exit(-1);
        }

        auto is_blank_line = [](const std::string &s) -> bool {
            return s.find_first_not_of(" \t\n\r") == std::string::npos;
        };

        std::vector<std::string> block;

        auto flush_block = [&]() {
            if (block.empty())
            {
                return;
            }

            // 现在只使用前两行，第三行如果存在则忽略。
            if (block.size() < 2)
            {
                std::cout << "Information error in: " << txt_path << std::endl;
                std::cout << "Each block should contain at least base_path and query_path." << std::endl;
                std::cout << "Current block size = " << block.size() << std::endl;
                exit(-1);
            }
            if (block.size() > 3)
            {
                std::cout << "Information warning in: " << txt_path << std::endl;
                std::cout << "A block contains more than 3 non-empty lines. "
                        << "Only the first two lines are used, and the rest are ignored." << std::endl;
            }

            base_path_list.push_back(block[0]);
            query_path_list.push_back(block[1]);

            block.clear();
        };

        while (std::getline(file, line))
        {
            if (is_blank_line(line))
            {
                flush_block();
            }
            else
            {
                block.push_back(line);
            }
        }
        flush_block();
        file.close();

        if (show_summary)
        {
            // [base_path_list] check
            std::cout << "### [base_path_list] check in ComponentLoad_multi::set_path_from_txt(), before load ### " << std::endl;
            for (int i = 0; i < base_path_list.size(); i++)
            {
                std::cout << base_path_list[i] << std::endl;
            }
            std::cout << "### [base_path_list] check end ### \n"
                    << std::endl;

            // [query_path_list] check
            std::cout << "### [query_path_list] check in ComponentLoad_multi::set_path_from_txt(), before load ### " << std::endl;
            for (int i = 0; i < query_path_list.size(); i++)
            {
                std::cout << query_path_list[i] << std::endl;
            }
            std::cout << "### [query_path_list] check end ### \n"
                    << std::endl;
        }
    }
}