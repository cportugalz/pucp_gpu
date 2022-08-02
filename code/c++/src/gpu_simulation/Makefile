# usage 
# make n=<number of data generated>

.SILENT:
all: 
	nvcc -std=c++11 gpu_main_simulation.cpp ../pucp_neutrino/cuda_probabilities.cu -I../pucp_neutrino -lcublas -lcusolver -o gpu_simulation

run:
	./gpu_simulation $(n)

.SILENT:
clean:
	rm gpu_simulation

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make all"
	echo "to compile all the code then you can run the program with the lines below according to your configuration:"
	echo "	make run n = <number of data to simulate>"
	echo "******************"