// Multiplicación de matrices para comparar la velocidad de corrida en cpu contra la corrida en paralelo usando cluster de GPU

#include <iostream>

using namespace std;


// Multiplicacion de matrices en GPU
__global__ void productoMatricialKernel(int* A, int* B, int* C, int N){

    float Cvalue = 0;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for(int k = 0; k < N; k++){
        Cvalue += A[N * i + k] * B[N * k + j];
    }

    C[N * i + j] = Cvalue;
}


// multiplicacion de matrices en CPU
void productoMatricial(int* A, int* B, int* C, int N){

    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            int column_row_sum = 0;
            for(int k = 0; k < N; k++){
                column_row_sum += A[N * i + k] * B[N * k + j];
            }

            C[N * i + j] = column_row_sum;
        }
    }

}


int main(){
    int N = 4;
    int* A = new int [N*N];
    int* B = new int [N*N];
    int* C = new int [N*N];

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[N * i + j] =  N * i + j + 1; 

            if (i == j) B[N * i + j] = -1; 
            else  B[N * i + j] = 0;

            C[N*i + j] = 0;

        }
    }  
    productoMatricial(A, B, C, N);

    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}