#include "estarfm.h"
#include <iostream>
#include <cstring> 
#include <string>
#include <fstream>
#include <filesystem>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <params_file> <cpu/gpu>" << std::endl;
        return 1;
    }

    // 读取参数文件路径
    std::string paramsFilePath = argv[1];
    std::string mode = argv[2]; // "cpu" or "gpu"

    // 从文件中读取参数
    int win_size;
	int class_num;
    float  M_err;
	float _nodata;
    std::string BufferIn0,BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut;
    
    if (!readParamsFromFile(paramsFilePath, win_size, class_num, M_err, _nodata,
        BufferIn0, BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut)) {
        std::cerr << "Error reading parameters from file." << std::endl;
        return 1;
    }

    // 根据用户选择执行CPU或GPU版本
    if (mode == "cpu") {
        ESTARFM_CPU(BufferIn0, BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut, win_size, class_num, M_err, _nodata);
        std::cout << "SATRFM算法执行完毕（CPU），图像保存为: " << BufferOut << std::endl;
    }
    else if (mode == "gpu") {
        ESTARFM_GPU(BufferIn0, BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut, win_size, class_num, M_err, _nodata);
        std::cout << "SATRFM算法执行完毕（GPU），图像保存为: " << BufferOut << std::endl;
    }
    else {
        std::cerr << "Invalid mode. Choose 'cpu' or 'gpu'." << std::endl;
        return 1;
    }

    return 0;
}