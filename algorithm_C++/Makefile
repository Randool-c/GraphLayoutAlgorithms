output: main.o io.o graph.o matrix.o stress.o consts.o fast.o sgd.o disjoint_set.o nicely.o maxmatch.o \
	weighted_interpolation.o
	g++ main.o io.o graph.o matrix.o stress.o consts.o fast.o sgd.o disjoint_set.o nicely.o maxmatch.o weighted_interpolation.o \
		-o output

main.o: main.cpp layout utils/io.h utils/graph.h utils/matrix.h
	g++ -c main.cpp

io.o: utils/io.cpp utils/io.h
	g++ -c utils/io.cpp

graph.o: utils/graph.cpp utils/matrix.h utils/graph.h
	g++ -c utils/graph.cpp

disjoint_set.o: utils/disjoint_set.cpp utils/disjoint_set.h
	g++ -c utils/disjoint_set.cpp

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

nicely.o: multiscale/nicely.h
	g++ -c multiscale/nicely.cpp

maxmatch.o: multiscale/maxmatch.h
	g++ -c multiscale/maxmatch.cpp

weighted_interpolation.o: multiscale/weighted_interpolation.h
	g++ -c multiscale/weighted_interpolation.cpp

.PHONY: clean
clean:
	-rm main.o io.o graph.o matrix.o stress.o consts.o output fast.o sgd.o disjoint_set.o nicely.o maxmatch.o \
		weighted_interpolation.o