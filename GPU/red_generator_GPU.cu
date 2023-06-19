/* Idea preliminar de codigo en Cuda/C++:

    Las redes estudiadas son redes compuestas por Q cliques (https://es.wikipedia.org/wiki/Clique) con tamaños n dados por una distribución de probabilidad fn. A su vez
    cada nodo puede ser unido mediante un "enlace de largo alcance" con a lo sumo un nodo en otro clique.

    La forma de representar la red va a ser con los siguientes arrays:

    -> clique_sizes = [1,2,1,5,3,3, ...] -> vector del tamaño de cada clique (size = Q) (por ahora son solo tamaños iguales)
    -> nro_clique = [0,1,1,2,3,3,3,3,3, ...] -> vector de nodos en donde cada elemento es el nro de clique al que pertenece (tiene tamaño N = sum(clique_sizes))
    -> tiene_largo_alcance = [1,0,1,1,0, 0 , ...] -> vector que dice si el nodo tiene o no enlace de largo alcance
    -> vecino_largo_alcance = [20,10,3,2,-1,...] -> vector en donde se guarda el numero de nodo al que esta conectado cada uno en otro clique (si es -1 no tiene enlace de largo alcance)

*/
#include "Aux_Functions_GPU.h"


class Red_GPU{
    public:
        Red_GPU(int Q, float gamma, int m){ // constructor para cliques iguales
            this -> Q = Q; // cantidad de cliques
            this -> gamma = gamma; // probabilidad de enlaces de largo alcance
            this -> n_max = m; // tamaño de los cliques (en este caso todos constantes)
            this -> N = Q * m; // cantidad de nodos en la red
            this -> clique_sizes = nullptr; // va a servir cuando los cliques cambien de tamaño
            cudaMalloc(&(this -> nro_clique), N * sizeof(int)); 
            cudaMalloc(&(this -> tiene_largo_alcance), N * sizeof(int));
            cudaMalloc(&(this -> vecino_largo_alcance), N* sizeof(int));

            // establezco los nodos que van a tener enlace de largo alcance y escribo el numero de clique de cada nodo
            node_initial_setting<<< N/m , m >>>(N, gamma, cuRand_seed, nro_clique, vecino_largo_alcance, tiene_largo_alcance);

            int cantidad_largo_alcance = thrust::reduce(thrust::device, tiene_largo_alcance, tiene_largo_alcance + N);

            // asigno vecinos de largo alcance
            asigno_largo_alcance(m, N, cantidad_largo_alcance, nro_clique, tiene_largo_alcance, vecino_largo_alcance);


        }

        ~Red_GPU(){
            if (clique_sizes != nullptr){   
                cudaFree(clique_sizes);
            }
            if (nro_clique != nullptr){
                cudaFree(nro_clique);
            }
            if (tiene_largo_alcance != nullptr){
                cudaFree(tiene_largo_alcance);
            }
            if (vecino_largo_alcance != nullptr){
                cudaFree(vecino_largo_alcance);
            }
        }


    //private:
        int Q; // cantidad de cliques
        int n_max; // maximo tamaño que puede tener un clique
        int N;
        float gamma;
        int* clique_sizes; // por ahora va a ser nullptr porque trabajo con tamaño constante
        int* nro_clique; 
        int* tiene_largo_alcance;
        int* vecino_largo_alcance;
};

int main (){

    int Q = 15;
    int m = 3;
    float gamma = 1.0;
    int N = Q * m;

    Red_GPU red_test (Q, gamma, m);

    int* nro_clique_Host = new int [N]; 
    int* tiene_largo_alcance_Host = new int [N];
    int* vecino_largo_alcance_Host = new int [N];

    cudaMemcpy(nro_clique_Host, red_test.nro_clique, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tiene_largo_alcance_Host, red_test.tiene_largo_alcance, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vecino_largo_alcance_Host, red_test.vecino_largo_alcance, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << endl << "numero de clique : " << endl;
    cout << "{";
    for (size_t i = 0; i < N-1; ++i)
    {
        cout << nro_clique_Host[i] << ", ";
    }
    cout << nro_clique_Host[N-1] << "}\n";

    cout << endl << "largo alcance : " << endl;
 
    cout << "{";
    for (size_t i = 0; i < N-1; ++i)
    {
        cout << tiene_largo_alcance_Host[i] << ", ";
    }
    cout << tiene_largo_alcance_Host[N-1] << "}\n";


    cout << endl << "vecinos largo alcance : " << endl;
 
    cout << "{";
    for (size_t i = 0; i < N-1; ++i)
    {
        cout << vecino_largo_alcance_Host[i] << ", ";
    }
    cout << vecino_largo_alcance_Host[N-1] << "}\n";

    delete [] nro_clique_Host;
    delete [] tiene_largo_alcance_Host;
    delete [] vecino_largo_alcance_Host;

    return 0;
}