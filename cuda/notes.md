g++ -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` testf.cpp -o testf`python3-config --extension-suffix` -L. ~/Projects/Vis/Main/algorithms/cuda/libf.so
nvcc --shared -Xcompiler -fPIC compute_stress.cu -o libf.so
