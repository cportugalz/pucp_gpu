# usage 
# make oscillation

.SILENT:
oscillation:
	nvcc -std=c++11 oscillation_event.cu ../pucp_neutrino/cuda_probabilities.cu  -I../pucp_neutrino -lcublas -o oscillation
	./oscillation

.SILENT:
clean:
	rm oscillation 

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make"
	echo "to compile all the code then you can run the program with the lines below according to your configuration:"
	echo "	make oscillation  
	echo "******************"