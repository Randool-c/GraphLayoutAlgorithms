#include"utils/matrix.hpp"
#include<vector>
using namespace std;


int main(){
    mat::Mat<float> xarr(3, 3);
    mat::random(xarr);
    vector<int> container = {2,3,4,5};
    xarr.print();

    xarr(1, 1) = container[1];
    xarr(2, 2) = container[1];
    xarr.print();
    xarr(1, 1) = xarr(2, 2);
    xarr(2, 2) = 101.111;
    xarr.print();

    // container[1] = 101010;
    // xarr.print();
    // printf("%d\n", container[1]);
    return 0;
}