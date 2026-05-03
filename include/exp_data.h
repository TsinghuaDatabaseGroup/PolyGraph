#ifndef EXP_DATA_H
#define EXP_DATA_H

#include <string.h>
#include <iostream>
#include <vector>

#include "parameters.h"



void VAMANA_PARA(std::string dataset, xmt::Parameters &parameters)
{
    unsigned L, R;

    if (dataset == "New")
    {
        L = 75, R = 70; 
    }
    else if (dataset == "ImageText")
    {
        L = 200, R = 40; 
    }
    else if (dataset == "QA" || dataset == "QA2")
    {
        L = 120, R = 80; 
    }
    else if (dataset == "Wiki")
    {
        L = 180, R = 120; 
    }
    else if (dataset == "Protein")
    {
        L = 240, R = 160; 
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }


    if (!parameters.exist("L")){
        parameters.set<unsigned>("L", R);
    }
    if (!parameters.exist("L_refine")){
        parameters.set<unsigned>("L_refine", L);
    }
    if (!parameters.exist("R_refine")){
        parameters.set<unsigned>("R_refine", R);
    }
}

void ORACLE_VAMANA_PARA(std::string dataset, xmt::Parameters &parameters)
{
    unsigned L, R;
    if (dataset == "New")
    {
        L = 75, R = 70; 
    }
    else if (dataset == "ImageText")
    {
        L = 225, R = 210; 
    }
    else if (dataset == "QA" || dataset == "QA2")
    {
        L = 1125, R = 1050; 
    }
    else if (dataset == "Wiki")
    {
        L = 4725, R = 4410; 
    }
    else if (dataset == "Protein")
    {
        L = 19125, R = 17850; 
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    
    
    if (!parameters.exist("L")){
        parameters.set<unsigned>("L", R);
    }
    if (!parameters.exist("L_refine")){
        parameters.set<unsigned>("L_refine", L);
    }
    if (!parameters.exist("R_refine")){
        parameters.set<unsigned>("R_refine", R);
    }
}


void HNSW_PARA(std::string dataset, xmt::Parameters &parameters)
{
    unsigned max_m, max_m0, ef_construction;
    
    if (dataset == "New")
    {
        max_m = 20, max_m0 = 40, ef_construction = 100; // 1-field参数
    }
    else if (dataset == "ImageText")
    {
        max_m = 40, max_m0 = 80, ef_construction = 200; // 2-field参数 --> ef_construction = L_refine
    }
    else if (dataset == "QA" || dataset == "QA2")
    {
        max_m = 80, max_m0 = 160, ef_construction = 120; // 4-field参数 --> ef_construction = L_refine
    }
    else if (dataset == "Wiki")
    {
        max_m = 120, max_m0 = 240, ef_construction = 180; // 6-field参数 --> ef_construction = L_refine
    }
    else if (dataset == "Protein")
    {
        max_m = 160, max_m0 = 320, ef_construction = 240;  // 8-field参数 --> ef_construction = L_refine
    }
    else
    {
        std::cout << "dataset error!\n";
        exit(-1);
    }
    parameters.set<unsigned>("max_m", max_m);
    parameters.set<unsigned>("max_m0", max_m0);
    parameters.set<unsigned>("ef_construction", ef_construction);
    parameters.set<int>("mult", -1);
}



void set_data_path_PARAM(std::string dataset, xmt::Parameters &parameters)
{
    // dataset root path
    std::string dataset_root = parameters.get<std::string>("dataset_root");
    std::string txt_path, weight_path;
    std::string groun_truth_path;
    if (dataset == "New")
    {
        txt_path = "../dataset/path_info_New.txt";
        weight_path = "../dataset/useWeight/useWeightEachQuery_New.txt";
    }
    else if (dataset == "ImageText") // 2-fields
    {
        txt_path = "../dataset/path_info_CC1MNorm_2field_100W.txt";
        weight_path = "../dataset/useWeight/useWeightEachQuery_2fields_IT.txt";
    }
    else if (dataset == "QA" || dataset == "QA2") // 4-fields
    {
        txt_path = "../dataset/path_info_LMSYSType2Norm_4field_100W.txt";
        weight_path = "../dataset/useWeight/useWeightEachQuery_4fields_QA.txt";
    }
    else if (dataset == "Wiki") // 6-fields
    {
        txt_path = "../dataset/path_info_EnwikiNorm_6field_100W.txt";
        weight_path = "../dataset/useWeight/useWeightEachQuery_6fields_Wiki.txt";
    }
    else if (dataset == "Protein") // 8-fields
    {
        txt_path = "../dataset/path_info_ProteinNorm_8field_100W.txt";
        weight_path = "../dataset/useWeight/useWeightEachQuery_8fields_Protein.txt";
    }
    else
    {
        std::cout << "dataset input error!\n";
        std::cout << "  now, dataset name is: " << dataset << std::endl;
        exit(-1);
    }
    parameters.set<std::string>("txt_path", txt_path);
    parameters.set<std::string>("weight_path", weight_path);
};

/**
 * 在 main.cpp中被调用；处理内容：
 *      1. 获取每个dataset对应的path_info.txt地址
 *      2. 如果是构建：对不同dataset设置R, L等超参数
 */
void set_para(std::string alg, std::string dataset, xmt::Parameters &parameters)
{
    std::cout << "  -- In set_para(): alg = " << alg << std::endl;

    set_data_path_PARAM(parameters.get<std::string>("dataset"), parameters);

    // 只有build需要设置参数；search等只需要index
    if (!(parameters.get<std::string>("exc_type") == "build")) { return; }

    if (alg == "hnsw_fusion")
    {
        HNSW_PARA(dataset, parameters);
    }
    else if (
        alg == "PolyGraph" || alg == "vamana_equNoTotal" || alg == "vamana_fusion" || alg == "vamana_allWeight" 
    )
    {
        VAMANA_PARA(dataset, parameters);
    } 
    else if (alg == "vamana_oracle")
    {
        ORACLE_VAMANA_PARA(dataset, parameters);
    }
    else
    {
        std::cout << "algorithm input error!\n";
        exit(-1);
    }
}


#endif // EXP_DATA_H
