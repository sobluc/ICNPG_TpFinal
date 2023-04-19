// Multiplicación de matrices para comparar la velocidad de corrida en cpu contra la corrida en paralelo usando cluster de GPU

#include <iostream>

using namespace std;




void productoMatricial(int* A, int* B, int N){
    int* C = new int [N*N];
    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            int column_row_sum = 0;
            for(int k = 0; k < N; k++){
                column_row_sum += A[N * i + k] * B[N * k + j];
            }

            C[N * i + j] = column_row_sum;
        }
    }

    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            if(j == 0) cout << "[" ;

            if(j + 1 == N) cout << C[N*i + j] << "]" << endl;            
            else  cout << C[N*i + j] << '\t';
            
 
        }
    }    

    delete[] C;
}


int main(){
    int N = 4;
    int* A = new int [N*N];
    int* B = new int [N*N];


    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[N * i + j] =  N * i + j + 1; 
        }
    }  

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if (i == j) B[N * i + j] = -1; 
            else  B[N * i + j] = 0;
        }
    }

    productoMatricial(A, B, N);

    delete [] A;
    delete [] B;
    return 0;
}