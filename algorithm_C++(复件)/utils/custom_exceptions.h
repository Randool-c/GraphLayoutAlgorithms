//
// Created by chenst on 2019/12/31.
//

#ifndef MULTILEVEL_STRESS_C___CUSTOM_EXCEPTIONS_HPP
#define MULTILEVEL_STRESS_C___CUSTOM_EXCEPTIONS_HPP

#include<exception>


class MatMultiplyError: public std::exception{
public:
    const char * what() const noexcept override{
        return "Error! Matrix Multiplication can not operated on these two matrix\n";
    }
};

class ShapeNotMatch: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! shape not matched!\n";
    }
};

class IndexOutOfBound: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! Index out of boundary!\n";
    }
};

class SetColumnError: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! Input vector should have the same size with the number of rows of the target matrix\n";
    }
};

class SetRowError: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! Input vector should have the same size with the number of columns of the target matrix\n";
    }
};

class NotFullyConnectedError: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! The input graph is not a fully connected graph!\n";
    }
};

#endif //MULTILEVEL_STRESS_C___CUSTOM_EXCEPTIONS_HPP
