// estarfm.cpp (combined)
#include "estarfm.h"
#include <iostream>
#include <cstring> // Make sure to include cstring for strcmp
#include "gdal_priv.h"
#include "gdalwarper.h"
#include <stdio.h>
#include <ctime>
#include "cuda_runtime.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <fstream>
#include <sstream>


// --- Function Implementations (ESTARFM_CPU, ESTARFM_GPU, etc.) ---
// (All your existing code for ESTARFM_CPU, Blending2_CPU, etc. goes here)

// 计算标准差
float Cstddve_CPU(float** a, int n, int width, int height) {
    float Cstddve = 0, sumx = 0, sumxx = 0;
    for (int i = 0; i < width * height; i++) {
        sumx += a[n][i];
        sumxx += a[n][i] * a[n][i];
    }
    Cstddve = sqrt(sumxx / (width * height) - (sumx / (width * height)) * (sumx / (width * height)));
    return Cstddve;
}

// 计算相关系数 (两个时相)
void limit_a_CalcuRela_CPU(float** BufferIn11, float** BufferIn22, float** BufferIn33, float** BufferIn44, float** BufferIn55, int Height, int Width, int Win_size1, float M_err, int BandNum, int current, std::vector<int>& location_p, float* r, float* threshold_d, int task_height, float _nodata) {

    for (int j = 0; j < Height; j++)
    {
        for (int i = 0; i < Width; i++)
        {
            float dx, dy;
            float sumx, sumy, sumxy, sumxx, sumyy;
            int num = 0;

            dx = 0, dy = 0;
            num = 0;
            sumyy = 0;
            sumx = 0;
            sumy = 0;
            sumxy = 0;
            sumxx = 0;

            for (int ii = 0; ii < BandNum - 1; ii++) {
                if (BufferIn11[ii][j * Width + i] == BufferIn11[ii + 1][j * Width + i] && BufferIn22[ii][j * Width + i] == BufferIn22[ii + 1][j * Width + i])
                    num++;
            }

            if (num != (BandNum - 1) || (BandNum == 1)) {
                for (int ii = 0; ii < BandNum; ii++) {
                    sumxy = sumxy + BufferIn11[ii][j * Width + i] * BufferIn22[ii][j * Width + i] + BufferIn33[ii][j * Width + i] * BufferIn44[ii][j * Width + i];
                    sumx = sumx + BufferIn11[ii][j * Width + i] + BufferIn33[ii][j * Width + i];
                    sumy = sumy + BufferIn22[ii][j * Width + i] + BufferIn44[ii][j * Width + i];
                    sumxx = sumxx + BufferIn11[ii][j * Width + i] * BufferIn11[ii][j * Width + i] + BufferIn33[ii][j * Width + i] * BufferIn33[ii][j * Width + i];
                    sumyy = sumyy + BufferIn22[ii][j * Width + i] * BufferIn22[ii][j * Width + i] + BufferIn44[ii][j * Width + i] * BufferIn44[ii][j * Width + i];
                }
                dx = sqrt(sumxx / (BandNum * 2) - (sumx / (BandNum * 2)) * (sumx / (BandNum * 2)));
                dy = sqrt(sumyy / (BandNum * 2) - (sumy / (BandNum * 2)) * (sumy / (BandNum * 2)));
                r[j * Width + i] = (sumxy / (BandNum * 2) - sumx * sumy / (BandNum * BandNum * 4)) / (dx * dy);

                if (BandNum == 1 && r[j * Width + i] > 0)
                    r[j * Width + i] = 1;
                if (BandNum == 1 && r[j * Width + i] < 0)
                    r[j * Width + i] = -1;
            }
            else {
                r[j * Width + i] = 1;
            }

            if (r[j * Width + i] != r[j * Width + i])
                r[j * Width + i] = 0;
        }
    }
}

// 融合 (两个时相)
void Blending2_CPU(float** BufferIn11, float** BufferIn22, float** BufferIn33, float** BufferIn44, float** BufferIn55, float** BufferOut, int Height, int Width, int Win_size1, float M_err, int BandNum, int current, std::vector<int>& location_p, float* r, float* threshold_d, int task_height, float _nodata) {

    for (int j = current; j < current + task_height; j++)
    {
        for (int i = 0; i < Width; i++)
        {
            int rmin, rmax, smin, smax;
            int r1, s1;
            int Result1 = 0, m = 0;
            int n1;
            float dy;
            float sum, weight_all, weight;
            float aa = 0;
            float pix_sum1, pix_sum2;
            double Aver11;
            double Aver22;
            float Average1[10], Average2[10], Average3[10], Average4[10];
            float d = 0;
            float sumx, sumy, sumxy, sumxx, sumyy;
            float T1_weight;
            float T2_weight;

            aa = 0;
            pix_sum1 = 0;
            pix_sum2 = 0;
            for (m = 0; m < 10; m++) {
                Average1[m] = 0;
                Average2[m] = 0;
                Average3[m] = 0;
                Average4[m] = 0;
            }

            for (m = 0; m < BandNum; m++) {
                pix_sum1 += BufferIn11[m][i + Width * j];
                pix_sum2 += BufferIn33[m][i + Width * j];
            }

            if (fabs(pix_sum1 - _nodata * BandNum) > 1e-6 && fabs(pix_sum2 - _nodata * BandNum) > 1e-6) {
                n1 = 0;
                weight_all = 0, weight = 0;
                sum = 0;
                sumx = 0;
                Aver11 = 0;
                Aver22 = 0;
                sumy = 0;
                sumxy = 0;
                sumxx = 0;
                sumyy = 0;

                if (i - Win_size1 / 2 <= 0)
                    rmin = 0;
                else
                    rmin = i - Win_size1 / 2;

                if (i + Win_size1 / 2 >= Width - 1)
                    rmax = Width - 1;
                else
                    rmax = i + Win_size1 / 2;

                if (j - Win_size1 / 2 <= 0)
                    smin = 0;
                else
                    smin = j - Win_size1 / 2;

                if (j + Win_size1 / 2 >= Height - 1)
                    smax = Height - 1;
                else
                    smax = j + Win_size1 / 2;

                for (r1 = rmin; r1 <= rmax; r1++) {
                    for (s1 = smin; s1 <= smax; s1++) {
                        Result1 = 0;
                        for (m = 0; m < BandNum; m++) {
                            if (fabs(BufferIn11[m][r1 + Width * s1] - BufferIn11[m][i + Width * j]) <= threshold_d[m] && fabs(BufferIn33[m][r1 + Width * s1] - BufferIn33[m][i + Width * j]) <= threshold_d[m + BandNum]) {
                                Result1++;
                            }
                            else
                                break;
                        }

                        if (Result1 == BandNum) {
                            location_p[n1] = r1 + Width * s1;
                            d = 1 + sqrt((float)((r1 - i) * (r1 - i) + (s1 - j) * (s1 - j))) / (float)(Win_size1 / 2);
                            weight = 1.0 / ((1.0 - r[r1 + Width * s1]) * d + 0.0000001);
                            for (m = 0; m < BandNum; m++) {
                                Average1[m] += (BufferIn55[m][r1 + Width * s1] - BufferIn22[m][r1 + Width * s1]) * weight;
                                Average2[m] += (BufferIn55[m][r1 + Width * s1] - BufferIn44[m][r1 + Width * s1]) * weight;
                                Average3[m] += BufferIn11[m][r1 + Width * s1] * weight;
                                Average4[m] += BufferIn33[m][r1 + Width * s1] * weight;
                            }
                            weight_all += weight;
                            n1++;
                        }
                    }
                }

                if (n1 > 5) {
                    for (m = 0; m < BandNum; m++) {
                        sumx = 0;
                        sumy = 0;
                        sumxy = 0;
                        sumxx = 0;
                        sumyy = 0;
                        Aver11 = 0;
                        Aver22 = 0;
                        for (int k_h = 0; k_h < n1; k_h++) {
                            sumxy = sumxy + BufferIn11[m][location_p[k_h]] * BufferIn22[m][location_p[k_h]] + BufferIn33[m][location_p[k_h]] * BufferIn44[m][location_p[k_h]];
                            sumx = sumx + BufferIn11[m][location_p[k_h]] + BufferIn33[m][location_p[k_h]];
                            sumy = sumy + BufferIn22[m][location_p[k_h]] + BufferIn44[m][location_p[k_h]];
                            sumxx = sumxx + BufferIn11[m][location_p[k_h]] * BufferIn11[m][location_p[k_h]] + BufferIn33[m][location_p[k_h]] * BufferIn33[m][location_p[k_h]];
                            sumyy = sumyy + BufferIn22[m][location_p[k_h]] * BufferIn22[m][location_p[k_h]] + BufferIn44[m][location_p[k_h]] * BufferIn44[m][location_p[k_h]];
                        }
                        dy = sqrt(sumyy / (n1 * 2) - (sumy / (n1 * 2)) * (sumy / (n1 * 2)));
                        if (dy > M_err) {
                            aa = (sumxy - sumx * sumy / (2 * n1)) / (sumyy - sumy * sumy / (n1 * 2));
                            if (aa > 5 || aa < 0) {
                                aa = 1;
                            }
                        }
                        else {
                            aa = 1.0;
                        }
                        for (r1 = rmin; r1 <= rmax; r1++) {
                            for (s1 = smin; s1 <= smax; s1++) {
                                Aver11 += BufferIn55[m][r1 + Width * s1] - BufferIn22[m][r1 + Width * s1];
                                Aver22 += BufferIn55[m][r1 + Width * s1] - BufferIn44[m][r1 + Width * s1];
                            }
                        }
                        Aver11 = fabs(Aver11) / ((float)((rmax - rmin + 1) * (smax - smin + 1))) + 0.0000000001;
                        Aver22 = fabs(Aver22) / ((float)((rmax - rmin + 1) * (smax - smin + 1))) + 0.0000000001;
                        T1_weight = 1.0 / Aver11 / (1.0 / Aver11 + 1.0 / Aver22);
                        T2_weight = 1.0 / Aver22 / (1.0 / Aver11 + 1.0 / Aver22);
                        BufferOut[m][j * Width + i] = (BufferIn11[m][j * Width + i] + aa * Average1[m] / weight_all) * T1_weight + (BufferIn33[m][j * Width + i] + aa * Average2[m] / weight_all) * T2_weight;
                        if (BufferOut[m][j * Width + i] < 0)
                            BufferOut[m][j * Width + i] = Average3[m] * T1_weight / weight_all + Average4[m] * T2_weight / weight_all;
                    }
                }
                else {
                    for (m = 0; m < BandNum; m++) {
                        Aver11 = 0;
                        Aver22 = 0;
                        for (r1 = rmin; r1 <= rmax; r1++) {
                            for (s1 = smin; s1 <= smax; s1++) {
                                Aver11 += BufferIn55[m][r1 + Width * s1] - BufferIn22[m][r1 + Width * s1];
                                Aver22 += BufferIn55[m][r1 + Width * s1] - BufferIn44[m][r1 + Width * s1];
                            }
                        }
                        Aver11 = fabs(Aver11) / ((float)((rmax - rmin + 1) * (smax - smin + 1))) + 0.0000000001;
                        Aver22 = fabs(Aver22) / ((float)((rmax - rmin + 1) * (smax - smin + 1))) + 0.0000000001;
                        T1_weight = 1 / Aver11 / (1 / Aver11 + 1 / Aver22);
                        T2_weight = 1 / Aver22 / (1 / Aver11 + 1 / Aver22);
                        BufferOut[m][j * Width + i] = BufferIn11[m][j * Width + i] * T1_weight + BufferIn33[m][j * Width + i] * T2_weight;
                    }
                }
            }
            else {
                for (m = 0; m < BandNum; m++) {
                    BufferOut[m][j * Width + i] = 0;
                }
            }
        }
    }

}

// 执行融合 (两个时相)
void runtest1_CPU(float** BufferIn11, float** BufferIn22, float** BufferIn33, float** BufferIn44, float** BufferIn55, float** BufferOut, int Height, int Width, int Win_size1, float M_err, int BandNum, float* std, int current, int task_height, float _nodata) {

    float* r = new float[Height * Width];
    std::vector<int> Location_P(Win_size1 * Win_size1);

    limit_a_CalcuRela_CPU(BufferIn11, BufferIn22, BufferIn33, BufferIn44, BufferIn55, Height, Width, Win_size1, M_err, BandNum, current, Location_P, r, std, task_height, _nodata);
    Blending2_CPU(BufferIn11, BufferIn22, BufferIn33, BufferIn44, BufferIn55, BufferOut, Height, Width, Win_size1, M_err, BandNum, current, Location_P, r, std, task_height, _nodata);

    delete[] r;
}

// 分块处理 (两个时相)
void runtest_CPU(float** BufferIn11, float** BufferIn22, float** BufferIn33, float** BufferIn44, float** BufferIn55, float** BufferOut, int Height, int Width, int Win_size1, float M_err, int num_class, int BandNum, float _nodata) {
    float* std = new float[BandNum * 2];
    for (int i = 0; i < BandNum; i++) {
        std[i] = Cstddve_CPU(BufferIn11, i, Width, Height) * 2.0 / num_class;
        std[i + BandNum] = Cstddve_CPU(BufferIn33, i, Width, Height) * 2.0 / num_class;
        std::cout << std[i] << "  " << std[i + BandNum] << "  ";
    }

    //int sub_height = 500; // 可根据内存调整
    int sub_height = Height / 4; // 根据总高度分成4块

    int kk = 0;
    int i, j;
    float** sub_BufferIn11, ** sub_BufferIn22, ** sub_BufferIn33, ** sub_BufferIn44, ** sub_BufferIn55, ** sub_out;
    for (int heiht_all = 0; heiht_all < Height; heiht_all += sub_height) {
        int task_start = kk * sub_height;
        int task_end;
        if ((kk + 1) * sub_height - Height <= 0)
            task_end = (kk + 1) * sub_height - 1;
        else
            task_end = Height - 1;
        int data_start, data_end;
        if (task_start - Win_size1 / 2 <= 0)
            data_start = 0;
        else
            data_start = task_start - Win_size1 / 2;
        if (task_end + Win_size1 / 2 >= Height - 1)
            data_end = Height - 1;
        else
            data_end = task_end + Win_size1 / 2;
        int data_height = data_end - data_start + 1;
        sub_BufferIn11 = (float**)malloc(BandNum * sizeof(float*));
        sub_BufferIn22 = (float**)malloc(BandNum * sizeof(float*));
        sub_BufferIn33 = (float**)malloc(BandNum * sizeof(float*));
        sub_BufferIn44 = (float**)malloc(BandNum * sizeof(float*));
        sub_BufferIn55 = (float**)malloc(BandNum * sizeof(float*));
        sub_out = (float**)malloc(BandNum * sizeof(float*));
        for (int b = 0; b < BandNum; b++) {
            sub_BufferIn11[b] = new float[data_height * Width];
            sub_BufferIn22[b] = new float[data_height * Width];
            sub_BufferIn33[b] = new float[data_height * Width];
            sub_BufferIn44[b] = new float[data_height * Width];
            sub_BufferIn55[b] = new float[data_height * Width];
            sub_out[b] = new float[data_height * Width];
        }
        int copy;
        for (int k = 0; k < BandNum; k++) {
            copy = 0;
            for (i = data_start; i <= data_end; i++) {
                for (j = 0; j < Width; j++) {
                    sub_BufferIn11[k][copy * Width + j] = BufferIn11[k][i * Width + j];
                    sub_BufferIn22[k][copy * Width + j] = BufferIn22[k][i * Width + j];
                    sub_BufferIn33[k][copy * Width + j] = BufferIn33[k][i * Width + j];
                    sub_BufferIn44[k][copy * Width + j] = BufferIn44[k][i * Width + j];
                    sub_BufferIn55[k][copy * Width + j] = BufferIn55[k][i * Width + j];
                }
                copy++;
            }
        }
        int current = task_start - data_start;
        int task_height = task_end - task_start + 1;
        runtest1_CPU(sub_BufferIn11, sub_BufferIn22, sub_BufferIn33, sub_BufferIn44, sub_BufferIn55, sub_out, data_height, Width, Win_size1, M_err, BandNum, std, current, task_height, _nodata);

        for (int k = 0; k < BandNum; k++) {
            current = task_start - data_start;
            for (int i = task_start; i <= task_end; i++) {
                for (int j = 0; j < Width; j++) {
                    BufferOut[k][i * Width + j] = sub_out[k][current * Width + j];
                }
                current++;
            }
        }
        for (int g = 0; g < BandNum; g++) {
            delete[]sub_BufferIn11[g];
            delete[]sub_BufferIn22[g];
            delete[]sub_BufferIn33[g];
            delete[]sub_BufferIn44[g];
            delete[]sub_BufferIn55[g];
            delete[]sub_out[g];
        }
        free(sub_BufferIn11);
        free(sub_BufferIn22);
        free(sub_BufferIn33);
        free(sub_BufferIn44);
        free(sub_BufferIn55);
        free(sub_out);
        kk++;
    }
    delete[] std;
}

// 主函数 (两个时相) CPU版本
void ESTARFM_CPU(const std::string& BufferIn0, const std::string& BufferIn1, const std::string& BufferIn2, const std::string& BufferIn3, const std::string& BufferIn4, const std::string& BufferOut, int win_size, int class_num, float M_err, float _nodata) {
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    GDALDataset* Landsat0 = (GDALDataset*)GDALOpen(BufferIn0.c_str(), GA_ReadOnly);
    int width, height, BandNum;
    width = Landsat0->GetRasterXSize();
    height = Landsat0->GetRasterYSize();
    BandNum = Landsat0->GetRasterCount();

    float** BufferLandsat_0 = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferLandsat_0[b] = new float[width * height];
    }

    for (int k = 0; k < BandNum; k++) {
        GDALRasterBand* hInBand1 = Landsat0->GetRasterBand(k + 1);
        hInBand1->RasterIO(GF_Read, 0, 0, width, height, BufferLandsat_0[k], width, height, GDT_Float32, 0, 0);
    }

    GDALDataset* MODIS0 = (GDALDataset*)GDALOpen(BufferIn1.c_str(), GA_ReadOnly);
    float** BufferModis_0 = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferModis_0[b] = new float[width * height];
    }

    for (int k = 0; k < BandNum; k++) {
        GDALRasterBand* hInBand1 = MODIS0->GetRasterBand(k + 1);
        hInBand1->RasterIO(GF_Read, 0, 0, width, height, BufferModis_0[k], width, height, GDT_Float32, 0, 0);
    }

    GDALDataset* Landsat1 = (GDALDataset*)GDALOpen(BufferIn2.c_str(), GA_ReadOnly);
    float** BufferLandsat_1 = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferLandsat_1[b] = new float[width * height];
    }

    for (int k = 0; k < BandNum; k++) {
        GDALRasterBand* hInBand1 = Landsat1->GetRasterBand(k + 1);
        hInBand1->RasterIO(GF_Read, 0, 0, width, height, BufferLandsat_1[k], width, height, GDT_Float32, 0, 0);
    }

    GDALDataset* MODIS1 = (GDALDataset*)GDALOpen(BufferIn3.c_str(), GA_ReadOnly);
    float** BufferModis_1 = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferModis_1[b] = new float[width * height];
    }

    for (int k = 0; k < BandNum; k++) {
        GDALRasterBand* hInBand1 = MODIS1->GetRasterBand(k + 1);
        hInBand1->RasterIO(GF_Read, 0, 0, width, height, BufferModis_1[k], width, height, GDT_Float32, 0, 0);
    }

    GDALDataset* MODIS2 = (GDALDataset*)GDALOpen(BufferIn4.c_str(), GA_ReadOnly);

    float** BufferModis_2 = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferModis_2[b] = new float[width * height];
    }

    for (int k = 0; k < BandNum; k++) {
        GDALRasterBand* hInBand1 = MODIS2->GetRasterBand(k + 1);
        hInBand1->RasterIO(GF_Read, 0, 0, width, height, BufferModis_2[k], width, height, GDT_Float32, 0, 0);
    }

    GDALDriver* poDriver = (GDALDriver*)GDALGetDriverByName("GTiff");
    GDALDataset* poDstDS = poDriver->Create(BufferOut.c_str(), width, height, BandNum, GDT_Float32, NULL);
    double* geos = new double[6];
    Landsat0->GetGeoTransform(geos);
    poDstDS->SetGeoTransform(geos);
    poDstDS->SetProjection(Landsat0->GetProjectionRef());

    float** BufferOutColor = new float* [BandNum];
    for (int b = 0; b < BandNum; b++) {
        BufferOutColor[b] = new float[width * height];
    }

    long now1 = clock();
    runtest_CPU(BufferLandsat_0, BufferModis_0, BufferLandsat_1, BufferModis_1, BufferModis_2, BufferOutColor, height, width, win_size, M_err, class_num, BandNum, _nodata);
    printf("CPU time: %dms\n", int(((double)(clock() - now1)) / CLOCKS_PER_SEC * 1000));

    for (int b = 0; b < BandNum; b++) {
        GDALRasterBand* poBand = poDstDS->GetRasterBand(b + 1);
        poBand->RasterIO(GF_Write, 0, 0, width, height, BufferOutColor[b], width, height, GDT_Float32, 0, 0);
    }

    GDALClose(Landsat0);
    GDALClose(MODIS0);
    GDALClose(Landsat1);
    GDALClose(MODIS1);
    GDALClose(MODIS2);
    GDALClose(poDstDS);

    for (int b = 0; b < BandNum; b++) {
        delete[]BufferLandsat_0[b];
        delete[]BufferModis_0[b];
        delete[]BufferLandsat_1[b];
        delete[]BufferModis_1[b];
        delete[]BufferModis_2[b];
        delete[]BufferOutColor[b];
    }
    delete[]BufferLandsat_0;
    delete[]BufferModis_0;
    delete[]BufferLandsat_1;
    delete[]BufferModis_1;
    delete[]BufferModis_2;
    delete[]BufferOutColor;
    delete[]geos;
}

// 读取参数文件
bool  readParamsFromFile(const std::string& filePath, int& win_size, int& class_num, float& M_err,
    float& _nodata, std::string& BufferIn0, std::string& BufferIn1,
    std::string& BufferIn2, std::string& BufferIn3, std::string& BufferIn4, std::string& BufferOut) {
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening parameter file." << std::endl;
        return false;
    }

    // 读取文件内容并解析
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            if (key == "win_size") win_size = std::stoi(value);
            else if (key == "class_num") class_num = std::stoi(value);
			else if (key == "M_err") M_err = std::stof(value);
			else if (key == "_nodata") _nodata = std::stof(value);
            else if (key == "landsatPath1") BufferIn0 = value;
            else if (key == "modisPath1") BufferIn1 = value;
            else if (key == "modisPath2") BufferIn2 = value;
            else if (key == "landsatPath2") BufferIn3 = value;
			else if (key == "landsatPath3") BufferIn4 = value;
			else if (key == "outputPath") BufferOut = value;
        }
    }
    return true;
}