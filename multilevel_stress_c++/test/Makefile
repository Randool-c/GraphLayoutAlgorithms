output: test.o
	g++ test.o -o output

test.o: test.cpp
	g++ -c test.cpp -o test.o

.PHONY: clean
clean:
	-rm output test.o