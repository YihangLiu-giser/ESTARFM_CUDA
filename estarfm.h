#ifndef ESTARFM_H
#define ESTARFM_H

#include <string>
#include "gdal.h"
#include <cmath>
#include "device_launch_parameters.h"
#include <vector>

// Function to perform EStaRFM fusion (user will select CPU or GPU version)
void ESTARFM_CPU(const std::string& BufferIn0, const std::string& BufferIn1, const std::string& BufferIn2, const std::string& BufferIn3, const std::string& BufferIn4, const std::string& BufferOut, int win_size, int class_num, float M_err, float _nodata);
void ESTARFM_GPU(const std::string& BufferIn0, const std::string& BufferIn1, const std::string& BufferIn2, const std::string& BufferIn3, const std::string& BufferIn4, const std::string& BufferOut, int win_size, int class_num, float M_err, float _nodata);

// 读取参数文件函数声明
bool readParamsFromFile(const std::string& filePath, int& win_size, int& class_num, float& M_err,
    float& _nodata, std::string& BufferIn0, std::string& BufferIn1,
    std::string& BufferIn2, std::string& BufferIn3, std::string& BufferIn4, std::string& BufferOut);

#endif