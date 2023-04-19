#include <iostream>

using namespace std;

void productoMatricial(int* A, int* B, int N){
    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            cout << endl;
        }
    }
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

    return 0;
}