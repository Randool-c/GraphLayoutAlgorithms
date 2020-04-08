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

#include "../utils/consts.h"

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
            double *array;
            int n;

            Array(int size);

            Array(double *start_pos, int size);

            double &operator[](int idx);
        };

        Array *arr;

        // methods
        Mat(int n_r=1, int n_c=1);

        Mat(const Mat &other);

        Mat(double *start_pos, int n_r, int n_c);

        ~Mat();

        void free_arr();

        bool row_outof_bound(int irow);

        bool col_outof_bound(int icol);

        Mat &operator=(const Mat &other);

        Mat copy();

        double &operator()(int i, int j);

        double &operator()(int i);

        Mat operator()(std::vector<int> &i, std::vector<int> &j);

        int size() const;

        void print();

        Mat reshape(int newr, int newc) const;

        double l2_norm();

        Mat operator[](int irow) const;

        Mat operator^(int n) const;

        Mat mm(const Mat &other);

        double dot(const Mat &other);

        double item() const;

        Mat get_row(int irow);

        Mat get_rows(std::vector<int> &irows);

        Mat get_col(int icol);

        Mat get_cols(std::vector<int> &icols);

        void set_row(int irow, Mat &target);

        void set_rows(std::vector<int> &irows, Mat &target);

        void set_col(int icol, Mat &target);

        void set_cols(std::vector<int> &icols, Mat &target);

        void save(const std::string &path);

//        Mat min(int axis=0);

//        Mat max(int axis=0);

        Mat argmin(int axis=0);

        Mat argmax(int axis=0);
    };


Mat operator+(const Mat &m1, const Mat &m2);
Mat operator+(const Mat &m, double num);
Mat operator+(double num, const Mat &m);
Mat operator-(const Mat &m1, const Mat &m2);
Mat operator-(const Mat &m, double num);
Mat operator-(double num, const Mat &m);
Mat operator*(const Mat &m1, const Mat &m2);
Mat operator*(const Mat &m, double num);
Mat operator*(double num, const Mat &m);
Mat operator/(const Mat &m1, const Mat &m2);
Mat operator/(const Mat &m, double num);
Mat operator/(double num, const Mat &m);
Mat mm(const Mat &m1, const Mat &m2);

void random(Mat &m);
void zeros(Mat &m);
void arange(Mat &m, int start=0);
Mat random(int nr, int nc);
Mat empty(int nr, int nc);
Mat zeros(int nr, int nc);

Mat fill(int nr, int nc, double number);
};

#endif //MULTILEVEL_STRESS_C___MATRIX_H
