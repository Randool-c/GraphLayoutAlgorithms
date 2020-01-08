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
#include<queue>
#include<string>
#include<cmath>
using namespace std;

int POS_INT_INF = 0x7f800000;
int NEG_INT_INF = 0xff800000;
const float POS_INF = *((float*)&POS_INT_INF);
const float NEG_INF = *((float*)&NEG_INT_INF);

class TestClass{
public:
    int *a;
    TestClass(){
        a = new int[10];
        a[2] = 12;
    }
    ~TestClass(){
        cout << "hello wrold" << endl;
//        delete a;
    }
    int &operator[](int idx){
        return a[idx];
    }
};

int main(){
    TestClass *tmp = new TestClass();
    if (tmp){
        cout << "allocated" << endl;
    }
    delete tmp;
//    cout << (*tmp)[2] << endl;
    if (!tmp){
        cout << "deleted" << endl;
    }
    return 0;
}
