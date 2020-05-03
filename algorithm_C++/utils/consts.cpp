#include "consts.h"

//int __POS_INF_FLAG = 0x7f800000, __NEG_INF_FLAG = 0xff800000;
//const double POS_INF = *((double *)&__POS_INF_FLAG);
//const double NEG_INF = *((double *)&__NEG_INF_FLAG);
const std::string ROOT = "/home/chenst/projects/Vis/Main/algorithms";
//
//__int64 __NaN=0xFFF8000000000000,__Infinity=0x7FF0000000000000,__Neg_Infinity=0xFFF0000000000000;
//const double NaN=*((double *)&__NaN),Infinity=*((double *)&__Infinity),Neg_Infinity=*((double *)&__Neg_Infinity);

long long __POS_INF_FLAG = 0x7FF0000000000000, __NEG_INF_FLAG = 0xFFF0000000000000;
const double POS_INF = *((double *)&__POS_INF_FLAG);
const double NEG_INF = *((double *)&__NEG_INF_FLAG);