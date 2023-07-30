#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>


class ImageProcess {
public:
    ImageProcess(){}
    void Set(int user_def = 0, float threshold = 0.5)
    {
        thresh_bool = user_def;
        thresh = threshold;
    }

    std::vector<std::vector<std::vector<float>>> Image3DSeg(std::vector<std::vector<std::vector<float>>>& NumpyArr, float porosity_u, float sigma) 
    {
        Img3DGray = NumpyArr;
        //std::cout << NumpyArr.size() << std::endl;
        //float porosity = porosity_u;
        int hist[256] = { 0 };
        for (int k = 0; k < Img3DGray.size(); k++)
        {
            for (int i = 0; i < Img3DGray.front().size(); i++)
            {
                for (int j = 0; j < Img3DGray.front().front().size(); j++)
                {
                    int t = Img3DGray[k][i][j] * 255;
                    hist[t]++;
                }
            }
        }
        srand((int)time(0));
        double rand_num = rand() / double(RAND_MAX);
        double tmp_thresh = 0;
        double minErr = 999999;
        for (int k = 0; k < 256; k++)
        {
            double sum0 = 0;
            for (int i = 0; i <= k; i++)
            {
                sum0 += hist[i];
            }
            double tmpErr = abs(1 - sum0 / (Img3DGray.size() * Img3DGray.front().size() * Img3DGray.front().front().size() * 1.0) - (porosity_u + sigma * (2 * rand_num - 1)));
            if (tmpErr < minErr)
            {
                minErr = tmpErr;
                tmp_thresh = k;
            }
        }
        if (thresh_bool)
            tmp_thresh = thresh;
        else 
            tmp_thresh = tmp_thresh / 255.0;
        std::cout << "3D segmentation threshold:" << tmp_thresh << std::endl;
        //std::cout << porosity * (1 + (2 * rand_num - 1) / 100) << std::endl;

        //Segment
        for (int z = 0; z < Img3DGray.size(); z++)
        {
            for (int y = 0; y < Img3DGray.front().size(); y++)
            {
                for (int x = 0; x < Img3DGray.front().front().size(); x++)
                {
                    if (Img3DGray[z][y][x] > tmp_thresh)
                    {
                        NumpyArr[z][y][x] = (float)1;
                    }
                    else
                    {
                        NumpyArr[z][y][x] = (float)0;
                    }
                }
            }
        }
        return  NumpyArr;
    }
private:
    std::vector<std::vector<std::vector<float>>> Img3DGray;
    int thresh_bool;
    float thresh;
};

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example";
    pybind11::class_<ImageProcess>(m, "GT")
        .def(pybind11::init())
        .def("set", &ImageProcess::Set)
        .def("img3dseg", &ImageProcess::Image3DSeg);
}