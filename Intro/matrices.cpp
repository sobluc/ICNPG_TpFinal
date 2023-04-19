// Multiplicación de matrices para comparar la velocidad de corrida en cpu contra la corrida en paralelo usando cluster de GPU

#include <iostream>

using namespace std;




void productoMatricial(int* A, int* B, int N){
    int* C = new int [N*N];
    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            int column_row_sum = 0;
            for(int k = 0; k < N; k++){
                column_row_sum += A[i + N*k] * B[k + N*j];
            }

            C[i + N*j] = column_row_sum;
        }
    }

    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            if(j == 0) cout << "[" ;

            if(j + 1 == N) cout << C[i + N*j] << "]" << endl;            
            else  cout << C[i + N*j] << '\t';
            
 
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
            A[i + N*j] =  i + N*j; 
        }
    }  

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if (i == j) B[i + N*j] = -1; 
            else  B[i + N*j] = 0;
        }
    }

    productoMatricial(A, B, N);

    delete [] A;
    delete [] B;
    return 0;
}