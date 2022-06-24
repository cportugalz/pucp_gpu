# Carpeta con funciones de probabilidad
LIB := Lib_Prob

########## Compilador SQuiDS en C++
LIBDIR:=lib
INCDIR:=include

GSL_CFLAGS=-I/usr/local/include
GSL_LDFLAGS=-L/usr/local/lib -lgsl -lgslcblas -lm

CFLAGS:= -Wno-abi -O3 -fPIC -I$(INCDIR) $(GSL_CFLAGS)
CXXFLAGS:= -std=c++11
LDFLAGS:= -Wl,-rpath -Wl,$(LIBDIR) -L$(LIBDIR) $(GSL_LDFLAGS)

#PATH_SQUIDS=../..
CXXFLAGS+=-I$(PATH_SQUIDS)/$(INCDIR)
LDFLAGS+= -Wl,-rpath -Wl,$(PATH_SQUIDS)/$(LIBDIR) -L$(PATH_SQUIDS)/$(LIBDIR) -lSQuIDS $(GSL_LDFLAGS)

######################

# Primero calculamos las Probabilidades, limpiamos los .o y ejecutamos
all:	Probabilidades	clean	

# Ejecución del programa principal y librerias. Sólo escribir 'make'+ Enter en la lína de comandos, luego ./Prob
Probabilidades :	Probabilidades.o	MyWrapperINV.o	MyWrapperVEP.o	MyWrapperNSI.o	MyWrapperVIS.o	MyWrapperDCH.o
	g++	Probabilidades.o	$(LIB)/MyWrapperINV.o	$(LIB)/MyWrapperVEP.o	$(LIB)/MyWrapperNSI.o	$(LIB)/MyWrapperVIS.o	$(LIB)/MyWrapperDCH.o	$(LDFLAGS)	-o	Prob
	
Probabilidades.o :	Probabilidades.c
	gcc	-c	Probabilidades.c	-o	Probabilidades.o	

MyWrapperINV.o :	$(LIB)/anth_MyWrapper_INV.cc
	g++	-c	$(LIB)/anth_MyWrapper_INV.cc	-o	$(LIB)/MyWrapperINV.o

MyWrapperVEP.o :	$(LIB)/anth_MyWrapper_VEP.cc
	g++	-c	$(LIB)/anth_MyWrapper_VEP.cc	-o	$(LIB)/MyWrapperVEP.o

MyWrapperNSI.o :	$(LIB)/anth_MyWrapper_NSI.cc
	g++	-c	$(LIB)/anth_MyWrapper_NSI.cc	-o	$(LIB)/MyWrapperNSI.o
	
MyWrapperVIS.o :	$(LIB)/anth_MyWrapper_VIS.cc
	g++	-c	$(LIB)/anth_MyWrapper_VIS.cc	-o	$(LIB)/MyWrapperVIS.o

MyWrapperDCH.o :	$(LIB)/anth_MyWrapper_DCH.cc
	g++	-c	$(LIB)/anth_MyWrapper_DCH.cc	-o	$(LIB)/MyWrapperDCH.o

clean: 
	rm	-f	Probabilidades.o	$(LIB)/MyWrapperINV.o	$(LIB)/MyWrapperVEP.o	$(LIB)/MyWrapperNSI.o	$(LIB)/MyWrapperVIS.o	$(LIB)/MyWrapperDCH.o
run:
	./Prob
