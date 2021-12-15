# usage 
# make single n=<number>
# make threaded n=<number> t=<number>

.SILENT:
all: 
	g++ -std=c++11 main_single.cpp Probabilities.cpp -o main_single
	g++ -std=c++11 main_threaded.cpp Probabilities.cpp -o main_threaded

.SILENT:
single:
	./main_single $(n)

.SILENT:
threaded:
	./main_threaded $(n) $(t)

.SILENT:
clean:
	rm main_single main_threaded *.txt
