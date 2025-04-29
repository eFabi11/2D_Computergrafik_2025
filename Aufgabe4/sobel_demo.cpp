#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


Eigen::MatrixXd sobel(Eigen::MatrixXd gray_img, Eigen::MatrixXd filter) {
    Eigen::MatrixXd filtered_img(gray_img.rows()-2, gray_img.cols()-2);
    for (int i = 0; i<gray_img.rows()-2; ++i) {
        for (int j = 0; j<gray_img.cols()-2; ++j) {
            int sum = 0;
            for (int fi = 0; fi < filter.rows(); ++fi) {
                for (int fj = 0; fj < filter.cols(); ++fj) {
                    sum += gray_img(i+fi,j+fj) * filter(fi,fj);
                }
            }
            filtered_img(i, j) = sum / 8.0;
        }
    }
    // TODO: implement filter operation

    return filtered_img;
}


PYBIND11_MODULE(sobel_demo, m) {
    m.doc() = "sobel operator using numpy!";
    m.def("sobel", &sobel);
}