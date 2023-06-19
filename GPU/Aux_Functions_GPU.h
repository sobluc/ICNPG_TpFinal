/* 
    Header con funciones auxiliares para la implementación de una red de cliques en CPU
*/

#include <ctime>
#include <iostream>
#include <cassert>

// cuRand
#include <curand.h>
#include <curand_kernel.h>

// thrust
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


using namespace std;

unsigned int cuRand_seed = time(0);


__global__ void node_initial_setting(int N, float gamma, unsigned int cuRand_seed,int* nro_clique, int* vecino_largo_alcance , int* tiene_largo_alcance){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N){
        nro_clique[i] = blockIdx.x;
        vecino_largo_alcance[i] = -1;

        curandState_t state;
        curand_init(cuRand_seed, i, 0, &state);
        if(curand_uniform(&state) <= gamma){
            tiene_largo_alcance[i] = 1;
        }
        else{
            tiene_largo_alcance[i] = 0;
        }
    }
}

__global__ void copio_si_tiene_largo_alcance(int N, int* tiene_largo_alcance, int* aux_con_largo_alcance){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N){
        if(tiene_largo_alcance[i] == 1){
            aux_con_largo_alcance[i] = i;
        }
        else{
            aux_con_largo_alcance[i] = -1;
        } 
    }

}


struct has_external_edge
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x != -1;
  }
};


__global__ void asigno_enlaces (int size, int* nodos_con_largo_alcance, int* vecino_largo_alcance, int* tiene_largo_alcance, int* nro_clique){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size){
        if(i % 2 == 0){
            
            if(i == size - 1){ // sirve si size es impar
                int nodo = nodos_con_largo_alcance[i];
                tiene_largo_alcance[nodo] = 0;
                vecino_largo_alcance[nodo] = -1;
            }
            else{
                int nodo0 = nodos_con_largo_alcance[i];
                int nodo1 = nodos_con_largo_alcance[i + 1];

                if(nro_clique[nodo0] != nro_clique[nodo1]){
                    vecino_largo_alcance[nodo0] = nodo1;
                    vecino_largo_alcance[nodo1] = nodo0;

                }
                else{
                    tiene_largo_alcance[nodo0] = 0;
                    tiene_largo_alcance[nodo1] = 1;

                    vecino_largo_alcance[nodo0] = -1;
                    vecino_largo_alcance[nodo1] = -1;
                
                }
            }
        }
    }
}


// asigno_largo_alcance asigna entre cuáles nodos va a existir un enlace de largo alcance
void asigno_largo_alcance(int m, int N, int cantidad_largo_alcance, int* nro_clique , int* tiene_largo_alcance , int* vecino_largo_alcance){
    int* aux_con_largo_alcance; 
    cudaMalloc(&aux_con_largo_alcance, N * sizeof(int));    

    copio_si_tiene_largo_alcance<<< N/m , m >>>(N, tiene_largo_alcance, aux_con_largo_alcance);

    thrust::device_vector<int> nodos_con_largo_alcance (cantidad_largo_alcance);
    thrust::copy_if(thrust::device, aux_con_largo_alcance, aux_con_largo_alcance + N, nodos_con_largo_alcance.begin(), has_external_edge());

    cudaFree(aux_con_largo_alcance);
    
    thrust::default_random_engine g;
    thrust::shuffle(thrust::device, nodos_con_largo_alcance.begin(), nodos_con_largo_alcance.end(), g);

    int* nodos_con_largo_alcance_raw_ptr = thrust::raw_pointer_cast(nodos_con_largo_alcance.data());

    int* mezclado = new int [cantidad_largo_alcance];

    cudaMemcpy(mezclado,  nodos_con_largo_alcance_raw_ptr , cantidad_largo_alcance * sizeof(int), cudaMemcpyDeviceToHost);
    cout << endl << "mezclado : " << endl;
    cout << "{";
    for (size_t i = 0; i < cantidad_largo_alcance-1; ++i)
    {
        cout << mezclado[i] << ", ";
    }
    cout << mezclado[cantidad_largo_alcance-1] << "}\n";

    int threadsPerBlock = 256;
    int totalBlocks = (cantidad_largo_alcance + (threadsPerBlock - 1))/threadsPerBlock;
    asigno_enlaces<<< totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(cantidad_largo_alcance, nodos_con_largo_alcance_raw_ptr, vecino_largo_alcance, tiene_largo_alcance, nro_clique);

}