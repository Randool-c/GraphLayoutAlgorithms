//
// Created by chenst on 2020/1/7.
//

#include "./matrix.h"
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


mat::Mat generate(){
    mat::Mat x = mat::random(5, 5);
    return x;
}


int main(){
    mat::Mat a = generate();
    vector<int> x = {1,2};
    vector<int> y = {2,4};
    a.print();
    mat::Mat b = a(x, y);
    cout << "yes " << endl;
    b.print();
    return 0;
}
