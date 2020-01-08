//
// Created by chenst on 2019/12/9.
//

#include<cstdio>
#include<iostream>
#include<vector>
#include<memory.h>
using namespace std;


int main(){
    int *a = new int [5];
    vector<int> b = {1,2,3,4,5};
    memcpy(a, &b[0], sizeof(int) * 5);
    cout << a[0] << endl;
    cout << a[1] << endl;
    cout << a[2] << endl;
    cout << a[3] << endl;
    cout << a[4] << endl;
    vector<int> c(a, a + 5);
    delete [] a;
    cout << c[1] << endl;

    return 0;
}
