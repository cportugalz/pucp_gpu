# usage 
# make n=<number of data generated>
CC=nvcc
CXXFLAGS= -std=c++11 -g
LIBS= -lcublas -lcusolver
PROGRAM=gpu_eigenv
SRCDIR=eigenvz
INCDIRS=-I$(SRCDIR)
OBJS = main.o \
	$(SRCDIR)/eigenv.o

# # Link c++ and CUDA compiled object files to target executable:
$(PROGRAM) : $(OBJS)
	$(CC) $(OBJS) -o $@  $(LIBS)

# Compile main .cpp file to object files:
%.o : %.cpp
	$(CC) -c $< -o $@

# # Compile C++ source files to object files:
# %.o : $(SRCDIR)/%.cpp $(SRCDIR)/%.h
# 	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

# # Compile C++ source files to object files:
# $(SRCDIR)/eigenv.o : $(SRCDIR)/eigenv.cu
# 	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@
$(SRCDIR)/eigenv.o : $(SRCDIR)/eigenv.cu
	$(CC) $(CXXFLAGS) $(INCDIRS) -c $< -o $@
	
run:
	./$(PROGRAM) $(n)

.SILENT:
clean:
	rm -f *.o $(PROGRAM)

help:
	echo "****** HELP ******"
	echo "Use:"
	echo "	make"
	echo "to compile all the code, then you can run the program with the lines below according to your configuration:"
	echo "	make run"
	echo "******************"