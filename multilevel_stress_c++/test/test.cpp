//
// Created by chenst on 2019/12/31.
//

#include<vector>
#include<algorithm>
#include <exception>
#include<iostream>
#include<memory.h>
#include<cstdlib>
using namespace std;

int POS_INT_INF = 0x7f800000;
int NEG_INT_INF = 0xff800000;
const float POS_INF = *((float*)&POS_INT_INF);
const float NEG_INF = *((float*)&NEG_INT_INF);

class MatCannotMultiply: public std::exception{
public:
    const char * what() const noexcept override{
        return "Error! Matrix Multiplication can not operated on these two matrix\n";
    }
};

class TestClass{
public:
    int a;
    int b;
    int *array;
    TestClass(int aa, int bb){
        a = aa;
        b = bb;
        array = new int[aa * bb];
    }

    int operator()(int i, int j){

    }
};

int f(int i, int j){
    if (i == 1) throw MatCannotMultiply();

    return i + j;
}

int main(){
//    int a = 1;
//    int *b = &a;
//    cout << f(2, 2) << endl;
    float a = 101001012913.12312;
    float d = 1.00001020120301231231239123120312;
    float q = 0.000000000000000000000000000000000000000112;
    float u = 0.0;
    double x[5] = {1.0};
    memset(x, 0, sizeof(double) * 5);
    x[2] += 1.12;
    for (double num : x){
        cout << num << endl;
    }
//    cout << (d / 0.0) << endl;
//    cout << INT32_MAX << endl;
    return 0;
}
