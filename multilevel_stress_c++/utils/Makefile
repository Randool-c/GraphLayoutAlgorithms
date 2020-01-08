#output: graph.o io.o consts.o test_io.o
#	g++ graph.o test_io.o io.o consts.o -o output
output: test_shortest_path.o matrix.o graph.o consts.o shortest_path.hpp
	g++ test_shortest_path.o matrix.o graph.o consts.o -o output

matrix.o: matrix.cpp
	g++ -c matrix.cpp

graph.o: graph.cpp graph.h
	g++ -c graph.cpp

io.o: io.cpp
	g++ -c io.cpp

consts.o: consts.cpp
	g++ -c consts.cpp

test_io.o: test_io.cpp
	g++ -c test_io.cpp

test_matrix.o: test_matrix.cpp matrix.hpp
	g++ -c test_matrix.cpp

clean:
	-rm test_io.o graph.o output consts.o io.o