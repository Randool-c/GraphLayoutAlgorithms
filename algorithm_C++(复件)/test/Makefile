output: testclass.o hello.o
	g++ testclass.cpp hello.o -o output

testclass.o: testclass.cpp testclass.h
	g++ -c testclass.cpp

hello.o: hello.cpp
	g++ -c hello.cpp

.PHONY: clean
clean:
	-rm output test.o testclass.o