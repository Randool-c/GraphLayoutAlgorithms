//
// Created by chenst on 2019/12/31.
//

//#ifndef MULTILEVEL_STRESS_C___MATRIX_HPP
//#define MULTILEVEL_STRESS_C___MATRIX_HPP

#include<vector>
#include<memory.h>
#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include <unistd.h>

#include"custom_exceptions.h"

#ifndef MULTILEVEL_STRESS_C___MATRIX_HPP
#define MULTILEVEL_STRESS_C___MATRIX_HPP

namespace mat {

//int TMP_POS_INT_INF = 0x7f800000;
//int TMP_NEG_INT_INF = 0xff800000;
//const float POS_INF = *((float*)&TMP_POS_INT_INF);
//const float NEG_INF = *((float*)&TMP_NEG_INT_INF);
//const float EPS = 1e-5;
//const int INT

/*
 * 实现时，取矩阵的某一行时，取出来的数据和原来的矩阵共享；但取列时，取出来的数据存在一个新的矩阵中，该矩阵和原矩阵没有关系
 * 保证Mat中的数据在内存中连续
 */
template<typename T> class Mat {
public:
    int nr;
    int nc;
    T *array;

    Mat(int n_r=1, int n_c=1): nr(n_r), nc(n_c) {
        array = (T *) malloc(sizeof(T) * n_r * n_c);
    }

    Mat(T *start_pos, int n_r, int n_c){
        array = start_pos;
        nr = n_r;
        nc = n_c;
    }

    Mat(Mat<T> &other){
        nr = other.nr;
        nc = other.nc;
        array = other.array;
    }

    Mat<T> copy(){
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i];
        }
        return ans;
    }

    void free(){
        free(array);
    }

    int size(){
        return nc * nr;
    }

    T &operator()(int i, int j){
        // fetch i-th row, j-th column element
        if (i < 0 || j < 0 || i >= nr || j >= nc){
            throw IndexOutOfBound();
        }
        return array[i * nc + j];
    }

    Mat<T> reshape(int newr, int newc){
        if (newr * newc != nr * nc){
            throw ShapeNotMatch();
        }
        return Mat(array, newr, newc);
    }

    Mat<T> operator*(Mat<T> const &other) const{
        if (nc != other.nc || nr != other.nr){
            throw ShapeNotMatch();
        }
        else{
            Mat<T> new_mat(nr, nc);
            for (int i = 0; i < nr * nc; ++i){
                new_mat.array[i] = array[i] * other.array[i];
            }
            return new_mat;
        }
    }

    Mat<T> operator*(T num){
        Mat<T> new_mat(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            new_mat.array[i] = array[i] * num;
        }
        return new_mat;
    }

    Mat<T> operator+(Mat<T> const &other) const{
        if (nc != other.nc || nr != other.nr){
            throw ShapeNotMatch();
        }
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i] + other.array[i];
        }
        return ans;
    }

    Mat<T> operator-(Mat<T> const &other) const{
        if (nc != other.nc || nr != other.nr){
            throw ShapeNotMatch();
        }
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i] - other.array[i];
        }
        return ans;
    }

    Mat<T> operator/(Mat<T> const &other) const{
        if (nc != other.nc || nr != other.nr){
            throw ShapeNotMatch();
        }

        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i] / other.array[i];
        }
        return ans;
    }

    Mat<T> mm(Mat<T> &other) const{
        if (nc != other.nr){
            throw MatMultiplyError();
        }
        else{
            Mat<T> ans(nr, other.nc);
            memset(ans.array, 0, sizeof(T) * nr * nc);
            for (int i = 0; i < nr; ++i) {
                for (int k = 0; k < nc; ++k) {
                    T tmp = array[i * nc + k];
                    for (int j = 0; j < other.nc; ++j) {
                        ans.array[i * other.nc + j] += tmp * other.array[k * other.nc + j];
                    }
                }
            }
            return ans;
        }
    }

    Mat<T> &operator=(Mat<T> const &other){
        if (this != &other) {
            nr = other.nr;
            nc = other.nc;
            array = other.array;
        }
        return *this;
    }

    Mat<T> get_row(int irow){
        if (irow < 0 || irow >= nr) throw IndexOutOfBound();

        T* new_array = (T*)malloc(sizeof(T) * nc);
        memcpy(new_array, array + irow * nc, sizeof(T) * nc);
        return Mat<T>(new_array, 1, nc);
    }

    Mat<T> get_col(int icol){
        if (icol < 0 || icol >= nc) throw IndexOutOfBound();

        T* new_array = (T*)malloc(sizeof(T) * nr);
        for (int i = 0; i < nr; ++i){
            new_array[i] = (*this)(i, icol);
        }
        Mat<T> tmp(new_array, nr, 1);
        return tmp;
    }

    Mat<T> operator[](int irow){
        // get the i-th row which shareds memory with the i-th row in the original matrix
        if (irow < 0 || irow >= nr) throw IndexOutOfBound();

        Mat<T> tmp(array + irow * nc, 1, nc);
        return tmp;
    }

    void set_row(int irow, Mat<T> &other){
        if (nc != other.size()) throw SetRowError();

        memcpy(array + irow * nc, other.array, sizeof(T) * nc);
    }

    void set_row(int irow, Mat<T> &other, int irow_other){
        // 将此矩阵的第irow行设置为另外一个矩阵的irow_other行

        if (nc != other.nc) throw SetRowError();

        memcpy(array + irow * nc, other.array + irow_other * nc, sizeof(T) * nc);
    }

    void set_row(int irow, T *input_array, int n_ele){
        if (n_ele != nc) throw SetRowError();

        memcpy(array + irow * nc, input_array, sizeof(T) * n_ele);
    }

    void set_col(int icol, Mat<T> &other){
        if (nr != other.size()) throw SetColumnError();

        for (int i = 0; i < other.size(); ++i) {
            array[i * nc + icol] = other.array[i];
        }
    }

    void set_col(int icol, Mat<T> &other, int icol_other){
        // 将此矩阵的第icol列设为另外一个矩阵的第icol_other列

        if (nr != other.nr) throw SetColumnError();

        for (int i = 0; i < other.nr; ++i){
            array[i * nc + icol] = other(i, icol_other);
        }
    }

    void set_col(int icol, T *input_array, int n_ele){
        if (nr != n_ele) throw SetColumnError();

        for (int i = 0; i < other.nr; ++i){
            array[i * nc + icol] = input_array[i];
        }
    }

    void print(){
        for (int i = 0; i < nr; ++i){
            for (int j = 0; j < nc; ++j){
                std::cout << array[i * nc + j] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const Mat<T> &m){
    os << m.nr << " " << m.nc << std::endl;
    for (int i = 0; i < m.nr; ++i){
        for (int j = 0; j < m.nc; ++j){
            os << m.array[i * m.nc + j] << '\t';
        }
        os << std::endl;
    }
}

void random(Mat<float> &matrix){
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < matrix.size(); ++i){
        matrix.array[i] = (float)(rand()) / RAND_MAX;
    }
}

Mat<float> zeros(Mat<float> &matrix){
    memset(matrix.array, 0, sizeof(float) * matrix.size());
}

}

#endif //MULTILEVEL_STRESS_C___MATRIX_HPP


//int main(){
//    //////////// Test Mat class /////////////
////    mat::Mat<float> a(3, 4);
////    mat::Mat<float> b(3, 4);
////    mat::random(a);
////    sleep(2);
////    mat::random(b);
//////    std::cout << a;
////    a.print();
////    b.print();
////
////    printf("a[2]\n");
//////    a[2].print();
////    printf("a[2]*2\n");
////    mat::Mat<float> tmpx;
////    tmpx = a[2] * 2;
//////    printf("%d %d\n", tmpx.nr, tmpx.nc);
////    (a[1] * b[1]).print();
////    (a / b).print();
//    mat::Mat<int> a(3, 4);
//    mat::Mat<int> b(3, 3);
//    for (int i = 1; i <= 12; ++i){
//        a.array[i - 1] = i;
//    }
//    for (int i = 1; i <= 30; ++i){
//        b.array[i - 1] = i;
//    }
//    a.print();
//    b.print();
////    (a.mm(b)).print();
//
////    a.set_col(1, b, 2);
//    mat::Mat<int> tmp;
//    tmp = b[2];
//    a.set_col(1, tmp);
//    a.print();
//    return 0;
//}
