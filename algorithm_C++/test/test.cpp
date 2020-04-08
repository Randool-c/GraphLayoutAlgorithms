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
#include<utility>
#include<set>
#include<map>
#include<string>
using namespace std;

int POS_INT_INF = 0x7f800000;
int NEG_INT_INF = 0xff800000;
const float POS_INF = *((float*)&POS_INT_INF);
const float NEG_INF = *((float*)&NEG_INT_INF);

//class TestClass{
//public:
//    int a;
//    TestClass(int x){
//        a = x;
//    }
//    TestClass(TestClass &other){
//        a = other.a;
//    }
//};
//
//TestClass operator@(TestClass &a, TestClass &b){
//    TestClass tmp;
//    tmp.a = a.a + b.a;
//    return tmp;
//}

void testfun(vector<string> &x){
    cout << x[0] << endl;
}

void printv(vector<pair<int, int>> &v){
    for (int i = 0; i < v.size(); ++i){
        printf(" (%d, %d) ", v[i].first, v[i].second);
    }
    printf("\n");
}

int main(){
    int *a = new int[10];
    for (int i = 0; i < 10; ++i) a[i] = i * 2;
    for (int x: a){
        cout << x << endl;
    }
    return 0;
}
