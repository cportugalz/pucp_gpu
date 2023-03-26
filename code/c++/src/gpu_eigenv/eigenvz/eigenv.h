#ifndef EIGENV_H
#define EIGENV_H

#include <cuComplex.h>

void eigen_system(int _batch, cuDoubleComplex* _input, double _error);

#endif