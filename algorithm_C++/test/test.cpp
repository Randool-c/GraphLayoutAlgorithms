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
    int a;
    TestClass(int x){
        a = x;
    }
    TestClass(TestClass &other){
        a = other.a;
    }
};

TestClass operator@(TestClass &a, TestClass &b){
    TestClass tmp;
    tmp.a = a.a + b.a;
    return tmp;
}

int main(){
    TestClass a(10);
    TestClass b(12);
    TestClass c;
    c = a + b;
    cout << c.a << endl;
    return 0;
}
