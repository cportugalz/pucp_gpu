# usage 
# make n=<number of data generated>
CC=gcc
LIBS= -lm
PROGRAM=eigenv
OBJS = eigenv.o	


# # Link c++ and CUDA compiled object files to target executable:
$(PROGRAM) : $(OBJS)
	$(CC) $(OBJS) -o $@ $(LIBS)

# Compile main .cpp file to object files:
%.o : %.c 
	$(CC) -c $< -o $@

run:
	./$(PROGRAM) 

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