//
// Created by chenst on 2020/4/16.
//

#include "b.h"


int main(){
    int N = 8;
    ClassA x = random(N * N);
    ClassA y = random(N * N);
    ClassA z = x.add(y);
    x.to_host();
    y.to_host();
    z.to_host();
    cout << "x: " << endl;
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            cout << x.data[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "y: " << endl;
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            cout << y.data[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "z: " << endl;
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            cout << z.data[i * N + j] << " ";
        }
        cout << endl;
    }
    return 1;
}
