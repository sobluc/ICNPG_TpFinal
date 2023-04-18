#include <iostream>

using namespace std;

void funcion(int** matriz, int N){


}


int main(){
    int N = 10
    int* A = new int*[N*N]

    cout << "veo si funciona" << endl;


    for(int i = 0, i < N, i++){
        for(int j = 0, j < N, j++){
            A[i + N*j] = i + N*j;
        }
    }  

    delete A[]

    return 0;
}