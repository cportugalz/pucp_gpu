# usage 
# make oscillation

.SILENT:
oscillation:
	g++ -std=c++11 oscillation_event.cpp ../pucp_neutrino/Probabilities.cpp ../pucp_neutrino/utils.cpp -I../pucp_neutrino -o oscillation
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