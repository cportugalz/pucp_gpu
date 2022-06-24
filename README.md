#  Cálculo de Matrices de Probabilidad - Version 0.2
Revisar la carpeta docs para mayor información de la teoría implementada aqui. Para esta revisión se ha agregado una función de integración con su respectiva biblioteca de código.

## Requerimientos  
1. Compilador de C++ 11.
2. Librería [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) de C++. Para instalar esta librería en Ubuntu 18.04, 20.X usar la siguiente linea de comando:
    ```bash
    sudo apt-get install libeigen3-dev
    ```
3. [Boost](https://www.boost.org/), puedes empezar con la [guía de instalación rápida.](https://www.boost.org/doc/libs/1_79_0/more/getting_started/unix-variants.html)

4. La probabilidad de Decoherencia requiere de la librería SQuIDS (intalador adjunto en code/src/SQuIDS). Descomprimir SQuIDS-master.zip y seguir las indicaciones (requisitos gsl y c++):
    1. Entrar a la carpeta SQuIDS-master
    2. Abrir un terminal en esa carpeta y ejecutar el comando: make
    3. Luego: make test
    4. Finalmente: sudo make install
    *** Si algo no funciona, revisar el archivo README.md en la carpeta SQuIDS-master

## Ejecución  

El código se divide en dos carpetas principales: C++ y Python(para implementar), en la carpeta C++ encontramos dos versiones del mismo código en las carpetas original y src, para ambos pueden seguir las instrucciones de su respectivo Makefile.

## Changelog
- Se añadió una nueva versión que incluye una función para el cálculo de ..

## Todo
- Añadir soporte GPU a las funciones de probabilidad.
- Añadir una versión en Python.
