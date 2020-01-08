//
// Created by chenst on 2020/1/7.
//

#include "matrix.h"
#include<iostream>
using namespace std;


//void printm(mat::Mat<float> const &input){
//    input.print();
//}

void hello(mat::Mat mat){
    cout << "refer2: " << mat.arr->refer << endl;
}

mat::Mat copy(){
    mat::Mat ans(10, 10);
    mat::Mat b = ans;

    cout << "refer: " << b.arr->refer << endl;
    hello(b);
    cout << "new refer: " << b.arr->refer << endl;
    return ans;
}


int main(){
    mat::Mat a(3, 4);
    mat::Mat b(4, 3);
    mat::Mat c;
    mat::arange(a);
    mat::arange(b);
    c = a.mm(b);
    a.print();
    b.print();
    c.print();
    return 0;
}
