# usage 
# make n=<number of data generated>

.SILENT:
all: 
	nvcc -std=c++11 -g gpu_main_simulation.cpp cuda_simulation.cpp ../pucp_neutrino/cuda_probabilities.cu ../pucp_neutrino/cuda_std_oscillation.cu ../pucp_neutrino/cuda_inv_decay.cu -I../pucp_neutrino -lcublas -lcusolver -o gpu_simulation

run:
	./gpu_simulation $(n)

.SILENT:
clean:
	rm gpu_simulation

check:
# valgrind --track-origins=yes  --gen-suppressions=all --leak-check=full --show-reachable=yes --suppressions=cudart.supp ./gpu_simulation $(n)
	valgrind --track-origins=yes --suppressions=cudart.supp ./gpu_simulation $(n)
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
	compute-sanitizer ./gpu_simulation $(n)

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make all"
	echo "to compile all the code then you can run the program with the lines below according to your configuration:"
	echo "	make run n = <number of data to simulate>"
	echo "******************"