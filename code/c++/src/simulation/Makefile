# usage 
# make single n=<number>
# make threaded n=<number> t=<number>

.SILENT:
all: 
	g++ -std=c++11 main_single.cpp ../pucp_neutrino/cpu/probabilities.cpp ../pucp_neutrino/cpu/utils.cpp -I../pucp_neutrino/cpu -I/home/cportugal/.local/include -lpthread -o main_single
	g++ -std=c++11 main_threaded.cpp ../pucp_neutrino/cpu/probabilities.cpp ../pucp_neutrino/cpu/utils.cpp -I../pucp_neutrino/cpu -I/home/cportugal/.local/include -lpthread -o main_threaded

.SILENT:
single:
	./main_single $(n)

.SILENT:
threaded:
	./main_threaded $(n) $(t)

.SILENT:
clean:
	rm main_single main_threaded output/*.txt

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make all"
	echo "to compile all the code then you can run the program with the lines below according to your configuration:"
	echo "	make single   n = <number of data to simulate>"
	echo "	make threaded n = <number od data to simulate> t = <number of cores to use>"
	echo "******************"