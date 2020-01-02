output: test_device.o
	nvcc test_device.o -o output

test_device.o: test_device.cu
	nvcc -c test_device.cu -o test_device.o -arch "sm_30"

.PHONY: clean
clean:
	-rm output test_device.o