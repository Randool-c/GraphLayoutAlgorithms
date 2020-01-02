//
// Created by chenst on 2019/12/31.
//

#include<vector>
#include<algorithm>
#include <exception>
#include<iostream>
#include<memory.h>
#include<cstdlib>
#include<unordered_map>
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

class TestArrayClass{
public:
    int *a;
    TestArrayClass(){
        a = new int[2];
        a[0] = 1;
        a[1] = 2;
    }
    int &operator[](int idx){
        return a[idx];
    }
};

int main(){
//    vector<int> a;
//    a.resize(10);
//    a.push_back(10101);
//    cout << a[10] << ' ' << a[0] << endl;
//    vector<unordered_map<int, int>> a(2);
//    vector<unordered_map<int, int>> b(3);
//    a[0].insert(make_pair(10, 10));
//    b[0].insert(make_pair(101, 101));
//    cout << a[0][10] << ' ' << b[0][101] << endl << endl;
//
//    a = b;
//    cout << a.size() << endl << endl;
//    cout << a[0][10] << ' ' << a[0][292] << ' ' << a[0][101] << endl;
//    a[0][101] = 1001010101;
//    cout << a[0][101] << ' ' << b[0][101] << endl;
    TestArrayClass a;
    cout << a[1] << endl;
    a[1] = 101010;
    cout << a[1] << endl;
    return 0;
}
