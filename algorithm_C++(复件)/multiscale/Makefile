output: kcenter.o graph.o matrix.o consts.o test_kcenter.o kcenter.o
	g++ kcenter.o graph.o matrix.o consts.o test_kcenter.o -o output

kcenter.o: fast/kcenter.cpp
	g++ -c kcenter.cpp

test_kcenter.o: test_kcenter.cpp
	g++ -c test_kcenter.cpp

graph.o: ../utils/graph.cpp
	g++ -c ../utils/graph.cpp

matrix.o: ../utils/matrix.cpp
	g++ -c ../utils/matrix.cpp

consts.o: ../utils/consts.cpp
	g++ -c ../utils/consts.cpp
