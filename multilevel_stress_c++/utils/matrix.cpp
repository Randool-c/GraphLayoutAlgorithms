//
// Created by chenst on 2020/1/8.
//

#include "matrix.h"

namespace mat {

    Mat::Array::Array(int size) {
        is_view = false;
        refer = 1;
        array = new float[size];
        n = size;
    }

    Mat::Array::Array(float *start_pos, int size) {
        is_view = true;
        refer = 1;
        array = start_pos;
        n = size;
    }

    float &Mat::Array::operator[](int idx) {
        return array[idx];
    }


    Mat::Mat(int n_r, int n_c) {
        nr = n_r;
        nc = n_c;
        arr = new Array(nr * nc);
    }

    Mat::Mat(float *start_pos, int n_r, int n_c) {
        nr = n_r;
        nc = n_c;
        arr = new Array(start_pos, nr * nc);
    }

    Mat::Mat(Mat &other) {
        nr = other.nr;
        nc = other.nc;
        arr = other.arr;
        arr->refer++;
    }

    Mat &Mat::operator=(Mat const &other) {
        if (this != &other) {
            nr = other.nr;
            nc = other.nc;
            arr = other.arr;
            arr->refer++;
        }
        return *this;
    }

    Mat::~Mat() {
        arr->refer--;
        if (arr->refer == 0) {
            if (!(arr->is_view)) {
                delete[] arr->array;
            }
            delete arr;
        }
    }

    bool Mat::row_outof_bound(int irow) {
        return irow < 0 || irow >= nr;
    }

    bool Mat::col_outof_bound(int icol) {
        return icol < 0 || icol >= nc;
    }

    Mat Mat::copy() {
        Mat ans(nr, nc);
        memcpy(ans.arr->array, arr->array, sizeof(float) * nr * nc);
        return ans;
    }

    float &Mat::operator()(int i, int j) {
        if (i < 0 || j < 0 || i >= nr || j >= nc) {
            throw IndexOutOfBound();
        }
        return arr->array[i * nc + j];
    }

    int Mat::size() {
        return nr * nc;
    }

    void Mat::print() {
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nc; ++j) {
                std::cout << arr->array[i * nc + j] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void Mat::reshape(int newr, int newc) {
        if (newr * newc != nr * nc) throw ShapeNotMatch();

        nr = newr;
        nc = newc;
    }

    Mat Mat::operator[](int irow) {
        if (irow < 0 || irow >= nr) throw IndexOutOfBound();
        Mat ans(arr->array + irow * nc, 1, nc);
        return ans;
    }

    Mat Mat::get_row(int irow) {
        if (row_outof_bound(irow)) throw IndexOutOfBound();

        Mat ans(nr, nc);
        memcpy(ans.arr->array, arr->array, sizeof(float) * nc * nr);
        return ans;
    }

    Mat Mat::get_col(int icol) {
        if (col_outof_bound(icol)) throw IndexOutOfBound();

        Mat ans(nr, nc);
        for (int i = 0; i < nr; ++i) {
            ans.arr->array[i] = arr->array[i * nc + icol];
        }
        return ans;
    }

    Mat operator+(const Mat &m1, const Mat &m2){
        if (m1.nc != m2.nc || m1.nr != m2.nr) throw ShapeNotMatch();

        Mat ans(m1.nr, m1.nc);
        for (int i = 0; i < m1.nr * m1.nc; ++i){
            ans.arr->array[i] = m1.arr->array[i] + m2.arr->array[i];
        }
        return ans;
    }

    Mat operator+(const Mat &m, float num){
        Mat ans(m.nr, m.nc);
        for (int i = 0; i < m.nr * m.nc; ++i){
            ans.arr->array[i] = num + m.arr->array[i];
        }
        return ans;
    }

    Mat operator+(float num, const Mat &m){
        Mat ans;
        ans = m + num;
        return ans;
    }

    Mat operator-(const Mat &m1, const Mat &m2){
        if (m1.nc != m2.nc || m1.nr != m2.nr) throw ShapeNotMatch();

        Mat ans(m1.nr, m1.nc);
        for (int i = 0; i < m1.nr * m1.nc; ++i) {
            ans.arr->array[i] = m1.arr->array[i] - m2.arr->array[i];
        }
        return ans;
    }

    Mat operator-(const Mat &m, float num){
        Mat ans;
        ans = m + (-num);
        return ans;
    }

    Mat operator-(float num, const Mat &m){
        Mat ans(m.nr, m.nc);
        for (int i = 0; i < m.nr * m.nc; ++i){
            ans.arr->array[i] = num - m.arr->array[i];
        }
        return ans;
    }

    Mat operator*(const Mat &m1, const Mat &m2) {
        if (m1.nc != m2.nc || m1.nr != m2.nr) throw ShapeNotMatch();

        Mat ans(m1.nr, m1.nc);
        for (int i = 0; i < m1.nr * m1.nc; ++i) {
            ans.arr->array[i] = m1.arr->array[i] * m2.arr->array[i];
        }
        return ans;
    }

    Mat operator*(const Mat &m, float num){
        Mat ans(m.nr, m.nc);
        for (int i = 0; i < m.nr * m.nc; ++i){
            ans.arr->array[i] = num * m.arr->array[i];
        }
        return ans;
    }

    Mat operator*(float num, const Mat &m){
        Mat ans;
        ans = m * num;
        return ans;
    }

    Mat operator/(const Mat &m1, const Mat &m2){
        if (m1.nc != m2.nc || m1.nr != m2.nr) throw ShapeNotMatch();

        Mat ans(m1.nr, m1.nc);
        for (int i = 0; i < m1.nr * m1.nc; ++i) {
            ans.arr->array[i] = m1.arr->array[i] / m2.arr->array[i];
        }
        return ans;
    }

    Mat operator/(float num, const Mat &m){
        Mat ans(m.nr, m.nc);
        for (int i = 0; i < m.nr * m.nc; ++i){
            ans.arr->array[i] = num / m.arr->array[i];
        }
        return ans;
    }

    Mat operator/(const Mat &m, float num){
        Mat ans;
        ans = m * (1 / num);
        return ans;
    }

    void random(Mat &m){
        srand(static_cast<unsigned int>(time(NULL)));
        for (int i = 0; i < m.size(); ++i){
            m.arr->array[i] = (float)(rand()) / RAND_MAX;
        }
    }

    void zeros(Mat &m){
        memset(m.arr->array, 0, sizeof(float) * m.size());
    }

    void arange(Mat &m, int start){
        for (int i = 0; i < m.size(); ++i){
            m.arr->array[i] = start + i;
        }
    }
}
