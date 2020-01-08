#include "consts.h"

int __POS_INF_FLAG = 0x7f800000, __NEG_INF_FLAG = 0xff800000;
const float POS_INF = *((float *)&__POS_INF_FLAG);
const float NEG_INF = *((float *)&__NEG_INF_FLAG);
const std::string ROOT = "/home/chenst/projects/Vis/Main/algorithms";
