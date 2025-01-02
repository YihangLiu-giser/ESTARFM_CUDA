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

    // ��ȡ�����ļ�·��
    std::string paramsFilePath = argv[1];
    std::string mode = argv[2]; // "cpu" or "gpu"

    // ���ļ��ж�ȡ����
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

    // �����û�ѡ��ִ��CPU��GPU�汾
    if (mode == "cpu") {
        ESTARFM_CPU(BufferIn0, BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut, win_size, class_num, M_err, _nodata);
        std::cout << "SATRFM�㷨ִ����ϣ�CPU����ͼ�񱣴�Ϊ: " << BufferOut << std::endl;
    }
    else if (mode == "gpu") {
        ESTARFM_GPU(BufferIn0, BufferIn1, BufferIn2, BufferIn3, BufferIn4, BufferOut, win_size, class_num, M_err, _nodata);
        std::cout << "SATRFM�㷨ִ����ϣ�GPU����ͼ�񱣴�Ϊ: " << BufferOut << std::endl;
    }
    else {
        std::cerr << "Invalid mode. Choose 'cpu' or 'gpu'." << std::endl;
        return 1;
    }

    return 0;
}