nvcc -c test_device.cu -o test_device.o -arch "sm_30"

nvcc test_device.o -o output

./output