# usage 
# make n=<number of data generated>
CC=nvcc
CXXFLAGS= -std=c++11 -g
LIBS= -lcublas -lcusolver
NEUTRINO=$(HOME)/pucp_gpu/code/c++/src
SRCDIR=$(NEUTRINO)/pucp_neutrino/gpu
LIBDIRS=
INCDIRS=-I$(NEUTRINO)/pucp_neutrino/gpu
PROGRAM=gpu_simulation
OBJS = gpu_simulation.o cuda_simulation.o \
		$(SRCDIR)/hutils.o \
		$(SRCDIR)/cuda_osc.o \
		$(SRCDIR)/cuda_inv.o \
		$(SRCDIR)/cuda_nsi.o \


# # Link c++ and CUDA compiled object files to target executable:
$(PROGRAM) : $(OBJS)
	$(CC) $(CXXFLAGS) $(OBJS) -o $@ $(INCDIRS) $(LIBS)

# Compile main .cpp file to object files:
%.o : %.cpp %.h $(SRCDIR)/%.h
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

# Compile C++ source files to object files:
%.o : $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

# Compile CUDA source files to object files:
$(SRCDIR)/cuda_osc.o : $(SRCDIR)/cuda_osc.cu
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(SRCDIR)/cuda_inv.o : $(SRCDIR)/cuda_inv.cu 
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@ 

$(SRCDIR)/cuda_nsi.o : $(SRCDIR)/cuda_nsi.cu
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@  
	
# .SILENT:
# all: $(objects)
# 	$(CC) $(CXXFLAGS) $(OBJS) $(INCDIRS) $(LIBS) -o $(PROGRAM)

run:
	./$(PROGRAM) $(n)

.SILENT:
clean:
	rm -f *.o $(SRCDIR)/*.o $(PROGRAM)

check:
# valgrind --track-origins=yes  --gen-suppressions=all --leak-check=full --show-reachable=yes --suppressions=cudart.supp ./gpu_simulation $(n)
	valgrind --track-origins=yes --suppressions=cudart.supp ./$(PROGRAM) $(n)
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
	compute-sanitizer ./$(PROGRAM) $(n)

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make"
	echo "to compile all the code then you can run the program with the lines below according to your configuration:"
	echo "	make run n = <number of data to simulate>"
	echo "******************"