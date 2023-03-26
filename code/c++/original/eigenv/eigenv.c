#include <stdio.h>
#include <math.h>
#include <complex.h> // Standard Library of Complex Numbers 
// Compilar con: gcc prueba_complejos.c -lm -o fe


double complex Determinante3D ( double complex A[3][3] )
{
return A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1] - A[0][2]*A[1][1]*A[2][0] - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2];
}

double complex Determinante2D ( double complex A[2][2] )
{
return A[0][0]*A[1][1] -  A[0][1]*A[1][0];
}

double ExpEntero ( double x , int n )
{ int i; double tmp = x;
  if ( n == 0) { return 1; }
  else if ( n > 0 ) {
  for ( i=1 ; i<n ; i++ ) { tmp = tmp*x; } return tmp; }
  else { printf("\n Exponente negativo. \n"); return 0; }
}

void CopiaMatriz ( double complex A[3][3], double complex B[3][3] )
{ int i, j; 
  for(i=0;i<3;i++)
  { for(j=0;j<3;j++)
	{ B[i][j] = A[i][j] ; } }
}

void MatrizReduc ( double complex A[3][3], int p, int q, double complex B[2][2] )
{ int i, j, c1=0, c2=0;
  for(i=0;i<3;i++) { c2 = 0;
   for(j=0;j<3;j++) { 
    if ( i != p && j != q ) { B[c1][c2] = A[i][j]; c2 = c2+1; }
     } if ( i != p ) { c1 = c1 + 1; } }
}

void MatrizAdjnt ( double complex A[3][3], double complex Adj[3][3] )
{ int i, j; double complex tmp[2][2];
  for(i=0;i<3;i++)
  { for(j=0;j<3;j++)
	{ MatrizReduc ( A, i, j, tmp );
	Adj[i][j] = ExpEntero( -1, i+j+2 ) * Determinante2D( tmp ) ; } }
}

void Transpuesta ( double complex A[3][3] )
{ double complex tmp[3][3]; CopiaMatriz ( A , tmp ); int i,j;
  for(i=0;i<3;i++)
  { for(j=0;j<3;j++)
	{ A[i][j] = tmp[j][i] ; } }
}

void MatrizInversa ( double complex A[3][3] , double complex Ainv[3][3] )
{ double complex detA = Determinante3D ( A ) ;
  double complex Adj[3][3]; MatrizAdjnt( A, Adj );
  Transpuesta ( Adj );
  int i,j;
  if ( detA != 0 ) {
  for(i=0;i<3;i++)
  { for(j=0;j<3;j++)
	{ Ainv[i][j] = Adj[i][j]/detA ; } } }
  else { printf("\n La matriz ingresda no tiene inversa. \n"); }
}

double complex px2 ( double complex x )
{ return x*x; }

double cpx2 ( double complex x )
{ return x*conj(x); }

double Norma ( double complex x[3] )
{ return sqrt( cpx2(x[0]) + cpx2(x[1]) + cpx2(x[2]) ); }

double complex MetodoPower ( double complex A[3][3], double complex x[3], double complex vect[3], double error )
{ int i,j,n=3; double complex x_new[3];
  double complex temp, lambda_new, lambda_old;
  
  	 // Inicializamos Lambda_Old 
	 lambda_old = 1;
 	 up1:
	 for(i=0;i<n;i++)
	 {
		  temp = 0.0;
		  for(j=0;j<n;j++)
		  {
		   	temp = temp + A[i][j]*x[j];
		  }
		  x_new[i] = temp;
	 }
	 // Reemplazando
	 for(i=0;i<n;i++)
	 {
	  	x[i] = x_new[i];
	 }
	 
	 // Encontrando el valor más grande 
	 lambda_new = x[0];
	 for(i=1;i<n;i++)
	 {
		  if(cabs(x[i])> cabs(lambda_new) ) 
		  {
		   	lambda_new = x[i];
		  }
	 }
	 // Normalizacion 
	 for(i=0;i<n;i++)
	 {
	  	x[i] = x[i]/lambda_new;
	 }

	 // Criterio de convergencia 
	 if( fabs( cabs(lambda_new) - cabs(lambda_old) ) > error  )
	 { 
		  lambda_old=lambda_new;
		  goto up1;
	 }

	 // Eigenvector
	 for(i=0;i<n;i++)
	 {
	  	vect[i] = x[i];
	 }
  
 return lambda_new; // Eigenvalor
}


void EigenSystem ( double complex A[3][3], double complex lambda[3], double complex vect[3][3], double error )
{ int i,j,k,n=3; double complex x[3][3]; double complex Atmp[3][3]; double complex Mtmp[3][3]; double complex MtmpInv[3][3];
  // Vector inicial
  for(i=0;i<n;i++)
  { for(j=0;j<n;j++)
	{ x[i][j] = j*0.1+1 ; }
  }
  
  // Diagonalización
  for ( k = 0 ; k < n  ; k++ )
  {
   if ( k == 0 ) { lambda[k] = MetodoPower ( A, x[k], vect[k], error ); }
   
   if ( k == 1 ) { 
   for(i=0;i<3;i++)
   { for(j=0;j<3;j++)
	{ Mtmp[i][j] = -( lambda[0]/px2( Norma(vect[0]) ) )*vect[0][i]*conj(vect[0][j]) ; 
	  Atmp[i][j] = A[i][j] + Mtmp[i][j] ;
	} } 
     lambda[k] = MetodoPower ( Atmp, x[k], vect[k], error );
     
   for(i=0;i<3;i++)
   { for(j=0;j<3;j++)
	{ Mtmp[i][j] = Mtmp[i][j]/lambda[k]; 
	  if ( i == j) { Mtmp[i][j] = Mtmp[i][j] + 1; } } }
	  
   MatrizInversa ( Mtmp , MtmpInv );
     
   for(i=0;i<3;i++)
   { vect[k][i] = MtmpInv[i][0]*x[k][0] + MtmpInv[i][1]*x[k][1] + MtmpInv[i][2]*x[k][2]; }
     
   }
   
  if ( k == 2 ) { 
   MatrizInversa ( A , MtmpInv );
   lambda[k] = (1 + 0*I)/MetodoPower ( MtmpInv, x[k], vect[k], error );
     
   }
  }
}


int main()
{
	 double error;
	 int i,j,n=3;
	 
	 double complex A[3][3]; double complex lambda[3]; double complex vect[3][3];
	 error = 1e-8; // Límite de error para convergencia
	 
	 // Matriz de Prueba
	 /*A[0][0] = 0.85 + 0.62*I; A[0][1] = 0.91 - 0.03*I; A[0][2] = 0.61;
	 A[1][0] = 0.065; A[1][1] = 0.51 + 0.98*I; A[1][2] = 0.1 + I;
	 A[2][0] = 0.78 - 0.002*I; A[2][1] = 0.951 - 0.369*I; A[2][2] = 0.55 + 0.72*I;*/
	 
	 // Matriz de Prueba
	 /*A[0][0] = 8.5 + 6.2*I; A[0][1] = 9.1 - 3*I; A[0][2] = 6.1;
	 A[1][0] = 6.5; A[1][1] = 5.1 ; A[1][2] = 1 + I;
	 A[2][0] = 7.8 - 2*I; A[2][1] = 9.51 - 3.69*I; A[2][2] = 5.5 + 7.2*I;*/
	 
	 // Matriz de Prueba
	 A[0][0] = 0.1-0.9*I; A[0][1] = 0.8-0.7*I; A[0][2] = 0.9+0.68*I;
	 A[1][0] = 0.7-0.23*I; A[1][1] = 0.5 + 0.12*I ; A[1][2] = 0.9 + 0.5*I ;
	 A[2][0] = 0.55+0.4*I; A[2][1] = -0.43+0.8*I; A[2][2] = 0.2-0.9*I;
	 
	 EigenSystem ( A, lambda , vect , error );
	 
	 printf("\n Eigen Valores: \n\n");
	 for(j=0;j<n;j++)
	 {
	  	printf("%f + %fi\t", creal(lambda[j]) , cimag(lambda[j]) );
	 }
	 
	 printf("\n\n Eigen vectores: \n\n");
	 for(i=0;i<n;i++)
	 {
	 for(j=0;j<n;j++)
	 {
	  	printf("%f + %fi\t", creal(vect[i][j]) , cimag(vect[i][j]) );
	 } printf("\n"); }
	 
	 
	 printf("\n\n\n");


	 
	 return(0);
}
