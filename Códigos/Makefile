# Carpeta con funciones de probabilidad
LIB := Lib_Prob

# Primero calculamos las Probabilidades, limpiamos los .o y ejecutamos
all:	Probabilidades	clean	run

# Ejecución del programa principal y librerias. Sólo escribir 'make'+ Enter en la lína de comandos, luego ./Prob
Probabilidades :	Probabilidades.o	MyWrapperINV.o	MyWrapperVEP.o	MyWrapperNSI.o
	g++	Probabilidades.o	$(LIB)/MyWrapperINV.o	$(LIB)/MyWrapperVEP.o	$(LIB)/MyWrapperNSI.o	-o	Prob
	
Probabilidades.o :	Probabilidades.c
	gcc	-c	Probabilidades.c	-o	Probabilidades.o	

MyWrapperINV.o :	$(LIB)/anth_MyWrapper_INV.cc
	g++	-c	$(LIB)/anth_MyWrapper_INV.cc	-o	$(LIB)/MyWrapperINV.o

MyWrapperVEP.o :	$(LIB)/anth_MyWrapper_VEP.cc
	g++	-c	$(LIB)/anth_MyWrapper_VEP.cc	-o	$(LIB)/MyWrapperVEP.o

MyWrapperNSI.o :	$(LIB)/anth_MyWrapper_NSI.cc
	g++	-c	$(LIB)/anth_MyWrapper_NSI.cc	-o	$(LIB)/MyWrapperNSI.o

clean: 
	rm	-f	Probabilidades.o	$(LIB)/MyWrapperINV.o	$(LIB)/MyWrapperVEP.o	$(LIB)/MyWrapperNSI.o
run:
	./Prob
