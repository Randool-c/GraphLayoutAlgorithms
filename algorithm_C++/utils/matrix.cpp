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

    Mat::Mat(const Mat &other) {
        nr = other.nr;
        nc = other.nc;
        arr = other.arr;
        arr->refer++;
    }

    Mat &Mat::operator=(const Mat &other) {
        if (this != &other) {
            free_arr();
            nr = other.nr;
            nc = other.nc;

            arr = other.arr;
            arr->refer++;
        }
        return *this;
    }

    Mat::~Mat() {
        free_arr();
    }

    void Mat::free_arr() {
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

    float &Mat::operator()(int i) {
        if (nr != 1 && nc != 1) throw ShapeNotMatch();
        else if (i < 0 || i >= nr * nc) throw IndexOutOfBound();
        else {
            return arr->array[i];
        }
    }

    Mat Mat::operator()(std::vector<int> &i, std::vector<int> &j){
        Mat ans(i.size(), j.size());
        for (int idx = 0; idx < i.size(); ++idx){
            for (int jdx = 0; jdx < j.size(); ++jdx){
                ans(idx, jdx) = arr->array[i[idx] * nc + j[jdx]];
            }
        }
        return ans;
    }

    int Mat::size() const {
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

    Mat Mat::reshape(int newr, int newc) const {
        Mat ans = *this;
        int size = nr * nc;
        if (newr == -1 && newc != -1){
            if (size % newc != 0) throw ShapeNotMatch();
            ans.nr = size / newc;
            ans.nc = newc;
        }
        else if (newr != -1 && newc == -1){
            if (size % newr != 0) throw ShapeNotMatch();
            ans.nc = size / newr;
            ans.nr = newr;
        }
        else if (newr * newc == nr * nc && nr != -1 && nc != -1){
            ans.nr = newr;
            ans.nc = newc;
        }
        else throw ShapeNotMatch();
        return ans;
    }

    Mat Mat::operator[](int irow) const {
        if (irow < 0 || irow >= nr) throw IndexOutOfBound();
        Mat ans(arr->array + irow * nc, 1, nc);
        return ans;
    }

    Mat Mat::operator^(int n) const {
        Mat ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.arr->array[i] = std::pow(arr->array[i], n);
        }
        return ans;
    }

    float Mat::l2_norm() {
        float sum_square = 0;
        for (int i = 0; i < nr * nc; ++i){
            sum_square += arr->array[i] * arr->array[i];
        }
        return std::sqrt(sum_square);
    }

    Mat Mat::mm(const mat::Mat &other) {
        if (other.nr == 1 && other.nc != 1){
            return mm(other.reshape(other.size(), 1));
        }
        else if (nc != other.nr){
            throw MatMultiplyError();
        }
        else{
            Mat ans(nr, other.nc);
//            memset(ans.arr->array, 0, sizeof(T) * nr * nc);
            zeros(ans);
            float tmp;
            for (int i = 0; i < nr; ++i) {
                for (int k = 0; k < nc; ++k) {
                    tmp = arr->array[i * nc + k];
                    for (int j = 0; j < other.nc; ++j) {
                        ans.arr->array[i * other.nc + j] += tmp * other.arr->array[k * other.nc + j];
                    }
                }
            }
            return ans;
        }
    }

    float Mat::dot(const mat::Mat &other) {
        if ((nr != 1 && nc != 1) || (other.nr != 1 && other.nc != 1) ||
            nr * nc != other.nr * other.nc) throw ShapeNotMatch();
        float ans= 0;
        for (int i = 0; i < nr * nc; ++i){
            ans += arr->array[i] * other.arr->array[i];
        }
        return ans;
    }

    float Mat::item() const{
        if (nr != 1 || nc != 1) throw ShapeNotMatch();
        return arr->array[0];
    }

    Mat Mat::get_row(int irow) {
        if (row_outof_bound(irow)) throw IndexOutOfBound();

        Mat ans(1, nc);
        memcpy(ans.arr->array, arr->array + irow * nc, sizeof(float) * nc);
        return ans;
    }

    Mat Mat::get_rows(std::vector<int> &irows){
        int n_rows = irows.size();
        Mat ans(n_rows, nc);
        for (int i = 0; i < n_rows; ++i){
            if (row_outof_bound(irows[i])) throw IndexOutOfBound();
//            for (int j = 0; j < nc; ++j){
//                ans(i, j) = (*this)(irows[i], j);
//            }
            memcpy(ans.arr->array + i * nc, arr->array + irows[i] * nc, sizeof(float) * nc);
        }
        return ans;
    }

    Mat Mat::get_col(int icol) {
        if (col_outof_bound(icol)) throw IndexOutOfBound();

        Mat ans(nr, 1);
        for (int i = 0; i < nr; ++i) {
            ans.arr->array[i] = arr->array[i * nc + icol];
        }
        return ans;
    }

    Mat Mat::get_cols(std::vector<int> &icols){
        int n_cols = icols.size();
        Mat ans(nr, n_cols);
        for (int j = 0; j < n_cols; ++j){
            if (col_outof_bound(icols[j])) throw IndexOutOfBound();
            for (int i = 0; i < nr; ++i){
                ans(i, j) = (*this)(i, icols[j]);
            }
        }
    }

    void Mat::set_row(int irow, mat::Mat &target) {
        if (nc != target.size()) throw SetRowError();

        memcpy(arr->array + irow * nc, target.arr->array, sizeof(float) * nc);
    }

    void Mat::set_rows(std::vector<int> &irows, mat::Mat &target) {
        if (target.nc != nc) throw SetRowError();

        for (int i = 0; i < irows.size(); ++i){
            if (row_outof_bound(irows[i])) throw IndexOutOfBound();

            memcpy(arr->array + irows[i] * nc, target.arr->array + i * nc, sizeof(float) * nc);
        }
    }

    void Mat::set_col(int icol, mat::Mat &target) {
        if (nr != target.size()) throw SetColumnError();

        for (int i = 0; i < target.size(); ++i){
            arr->array[i * nc + icol] = target.arr->array[i];
        }
    }

    void Mat::set_cols(std::vector<int> &icols, mat::Mat &target) {
        if (target.nr != nr) throw SetColumnError();

        for (int j = 0; j < icols.size(); ++j){
            if (col_outof_bound(icols[j])) throw IndexOutOfBound();

            for (int i = 0; i < nr; ++i){
                (*this)(i, icols[j]) = target(i, j);
            }
        }
    }

    void Mat::save(const std::string &path) {
        freopen((char*)path.data(), "w", stdout);
        for (int i = 0; i < nr; ++i){
            for (int j = 0; j < nc; ++j){
                std::cout << (*this)(i, j) << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        fclose(stdout);
    }

    Mat Mat::argmin(int axis) {
        float minvalue;
        float minpos;
        if (axis == 0){
            Mat ans(nr, 1);
            for (int i = 0; i < nr; ++i){
                minvalue = POS_INF;
                for (int j = 0; j < nc; ++j){
                    if (arr->array[i * nc + j] < minvalue){
                        minpos = j;
                        minvalue = arr->array[i * nc + j];
                    }
                }
                ans(i) = minpos;
            }
            return ans;
        }
        else if (axis == 1){
            Mat ans(1, nc);
            for (int i = 0; i < nc; ++i){
                minvalue = POS_INF;
                for (int j = 0; j < nr; ++j){
                    if (arr->array[j * nc + i] < minvalue){
                        minpos = j;
                        minvalue = arr->array[j * nc + i];
                    }
                }
                ans(i) = minpos;
            }
            return ans;
        }
    }

    Mat Mat::argmax(int axis) {
        float maxvalue;
        float maxpos;
        if (axis == 0){
            Mat ans(nr, 1);
            for (int i = 0; i < nr; ++i){
                maxvalue = NEG_INF;
                for (int j = 0; j < nc; ++j){
                    if (arr->array[i * nc + j] > maxvalue){
                        maxpos = j;
                        maxvalue = arr->array[i * nc + j];
                    }
                }
                ans(i) = maxpos;
            }
            return ans;
        }
        else{
            Mat ans(1, nc);
            for (int i = 0; i < nc; ++i){
                maxvalue = NEG_INF;
                for (int j = 0; j < nr; ++j){
                    if (arr->array[j * nc + i] > maxvalue){
                        maxvalue = arr->array[j * nc + i];
                        maxpos = j;
                    }
                }
                ans(i) = maxpos;
            }
            return ans;
        }
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

    Mat mm(const Mat &m1, const Mat &m2){
        if (m2.nr == 1 && m2.nc != 1){
            return mm(m1, m2.reshape(m2.size(), 1));
        }
        else if (m1.nc != m2.nr){
            throw MatMultiplyError();
        }
        else{
            Mat ans(m1.nr, m2.nc);
            zeros(ans);
            for (int i = 0; i < m1.nr; ++i) {
                for (int k = 0; k < m1.nc; ++k) {
                    float tmp = m1.arr->array[i * m1.nc + k];
                    for (int j = 0; j < m2.nc; ++j) {
                        ans.arr->array[i * m2.nc + j] += tmp * m2.arr->array[k * m2.nc + j];
                    }
                }
            }
            return ans;
        }
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

    void fill(Mat &m, float number){
        for (int i = 0; i < m.size(); ++i){
            m.arr->array[i] = number;
        }
    }

    Mat random(int nr, int nc){
        Mat ans(nr, nc);
        srand(static_cast<unsigned int>(time(NULL)));
        for (int i = 0; i < ans.size(); ++i){
            ans.arr->array[i] = (float)(rand()) / RAND_MAX;
        }
        return ans;
    }

    Mat empty(int nr, int nc){
        return Mat(nr, nc);
    }

    Mat zeros(int nr, int nc){
        Mat ans(nr, nc);
        mat::zeros(ans);
        return ans;
    }

    Mat fill(int nr, int nc, float number){
        Mat ans(nr, nc);
        fill(ans, number);
        return ans;
    }
}
