//
// Created by chenst on 2020/1/8.
//

#ifndef MULTILEVEL_STRESS_C___MATRIX_H
#define MULTILEVEL_STRESS_C___MATRIX_H

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

    class Mat {
    public:
        // members
        int nr;
        int nc;

        class Array {
            /*
             * 限制条件：当作为视图时，其所引用的对象若被销毁，那么该对象也将不可用;
             * 所有的赋值和复制构造函数都仅仅是指针的赋值，设计动态分配的内存其赋值后的指针指向同样的内存地址
             */
        public:
            int refer;      // 当refer降为0时，在外部类销毁时会同时销毁外部类中动态分配的Array内部类对象，否则不销毁
            bool is_view;   // 是不是与其他对象共享了一段数组, 作为一个视图. 作为视图时，销毁对象并不删除array的动态分配的内存.
            float *array;
            int n;

            Array(int size);

            Array(float *start_pos, int size);

            float &operator[](int idx);
        };

        Array *arr;

        // methods
        Mat() {}

        Mat(int n_r, int n_c);

        Mat(Mat &other);

        Mat(float *start_pos, int n_r, int n_c);

        ~Mat();

        bool row_outof_bound(int irow);

        bool col_outof_bound(int icol);

        Mat &operator=(Mat const &other);

        Mat copy();

        float &operator()(int i, int j);

        int size();

        void print();

        void reshape(int newr, int newc);

        float l2_norm();

        Mat operator[](int irow);

        Mat operator^(int n);

        Mat mm(const Mat &other);

        Mat get_row(int irow);

        Mat get_col(int icol);
    };


Mat operator+(const Mat &m1, const Mat &m2);
Mat operator+(const Mat &m, float num);
Mat operator+(float num, const Mat &m);
Mat operator-(const Mat &m1, const Mat &m2);
Mat operator-(const Mat &m, float num);
Mat operator-(float num, const Mat &m);
Mat operator*(const Mat &m1, const Mat &m2);
Mat operator*(const Mat &m, float num);
Mat operator*(float num, const Mat &m);
Mat operator/(const Mat &m1, const Mat &m2);
Mat operator/(const Mat &m, float num);
Mat operator/(float num, const Mat m);
Mat mm(const Mat &m1, const Mat &m2);

void random(Mat &m);
void zeros(Mat &m);
void arange(Mat &m, int start=0);
};

#endif //MULTILEVEL_STRESS_C___MATRIX_H
