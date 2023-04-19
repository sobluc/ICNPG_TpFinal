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
            
            cout << C[i + N*j] << '\t';
            
            if(j + 1 == N) cout << "]" << endl; 
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
            A[i + N*j] = i + N*j; 
        }
    }  

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            B[i + N*j] = -1; 
        }
    }

    

    delete [] A;
    delete [] B;
    return 0;
}