output: main.o io.o graph.o matrix.o stress.o consts.o fast.o sgd.o
	g++ main.o io.o graph.o matrix.o stress.o consts.o fast.o sgd.o -o output

main.o: main.cpp layout utils/io.h utils/graph.h utils/matrix.h
	g++ -c main.cpp

io.o: utils/io.cpp utils/io.h
	g++ -c utils/io.cpp

graph.o: utils/graph.cpp utils/matrix.h utils/graph.h
	g++ -c utils/graph.cpp

matrix.o: utils/matrix.h utils/matrix.cpp utils/custom_exceptions.h
	g++ -c utils/matrix.cpp

stress.o: layout/stress_majorization/stress.h utils/consts.h utils/matrix.h \
 			utils/shortest_path.hpp utils/graph.h layout/base_optimizer.hpp
	g++ -c layout/stress_majorization/stress.cpp

consts.o: utils/consts.cpp utils/consts.h
	g++ -c utils/consts.cpp

fast.o: multiscale/fast.h
	g++ -c multiscale/fast.cpp

sgd.o: layout/sgd/sgd.h
	g++ -c layout/sgd/sgd.cpp

.PHONY: clean
clean:
	-rm main.o io.o graph.o matrix.o stress.o consts.o output fast.o sgd.o