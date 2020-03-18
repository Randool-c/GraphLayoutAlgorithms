//
// Created by chenst on 2019/12/31.
//

#ifndef MULTILEVEL_STRESS_C___MATRIX_HPP
#define MULTILEVEL_STRESS_C___MATRIX_HPP

#include<vector>
#include<memory.h>
#include<cstdlib>
#include<algorithm>
#include<iostream>
#include<ctime>
#include<cstdio>
#include<stdlib.h>
#include <unistd.h>
#include <cmath>

#include"custom_exceptions.h"

namespace mat {

template<typename T=float> class Mat {
public:
    int nr;
    int nc;
    class Array{
        int reference;
        T *array;

        Array(){

        }
    };
    Array *array;

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

    ~Mat(){

    }

    Mat<T> &operator=(Mat<T> const &other){
        // 非赋值，而是存储的数据指向相同的内存地址
        if (this != &other) {
            nr = other.nr;
            nc = other.nc;
            if (array != other.array){
                delete array;
            }
            array = other.array;
        }
        return *this;
    }


    Mat<T> copy(){
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i];
        }
        return ans;
    }

    void free_mat(){
        delete array;
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

    Mat<T> operator-(T other) const{
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = other - array[i];
        }
        return ans;
    }

    Mat<T> operator-() const {
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = -array[i];
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

    Mat<T> square() const{
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = array[i] * array[i];
        }
        return ans;
    }

    Mat<T> reciprocal() const{
        // 计算 1 / x
        Mat<T> ans(nr, nc);
        for (int i = 0; i < nr * nc; ++i){
            ans.array[i] = 1 / array[i];
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

        for (int i = 0; i < n_ele; ++i){
            array[i * nc + icol] = input_array[i];
        }
    }

    float l2_norm(){
        if (nc != 1 && nr != 1){
            throw ShapeNotMatch();
        }
        else{
            float ans = 0;
            for (int i = 0; i < nr * nc; ++i){
                ans += array[i] * array[i];
            }
            return std::sqrt(ans);
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

inline void random(Mat<float> &matrix){
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < matrix.size(); ++i){
        matrix.array[i] = (float)(rand()) / RAND_MAX;
    }
}

template<typename T=float, typename T1=float>
Mat<T> operator-(T1 num, Mat<T> &target){
    Mat<T> ans(target.nr, target.nc);
    for (int i = 0; i < target.size(); ++i){
        ans.array[i] = num - target.array[i];
    }
    return ans;
}

template<typename T1, typename T2>
Mat<T2> operator/(T1 num, Mat<T2> &target){
    Mat<T2> ans(target.nr, target.nc);
    for (int i = 0; i < target.size(); ++i){
        ans.array[i] = num / target.array[i];
    }
    return ans;
}

template<typename T>
Mat<T> zeros(Mat<T> &matrix){
    memset(matrix.array, 0, sizeof(T) * matrix.size());
}


}

#endif //MULTILEVEL_STRESS_C___MATRIX_HPP
