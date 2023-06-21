#include "Aux_Functions_GPU.h"
#include "cpu_timer.h"


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


        int Q; 
        int n_max; 
        int N;
        float gamma;
        int* clique_sizes;
        int* nro_clique; 
        int* tiene_largo_alcance;
        int* vecino_largo_alcance;
};

int main (){

    // Asigno valores de Q (cantidad de cliques), m (tamaño de los cliques) y gamma (probabilidad de largo alcance)
    int Q = 10;
    int m = 3;
    float gamma = 1.0;

    cout << "Q : " << Q << "| m : " << m << " | gamma : " << gamma << endl;

    cpu_timer relojcpu;
    relojcpu.tic();
    Red_GPU red_test (Q, gamma, m); // creo la red con su constructor
    cout << relojcpu.tac() << endl;

    // Las siguientes lineas comentadas son para ver los vectores de la red creada (se recomienda usar con redes chicas)

    // int N = m*Q;

    // int* nro_clique_Host = new int [N]; 
    // int* tiene_largo_alcance_Host = new int [N];
    // int* vecino_largo_alcance_Host = new int [N];

    // cudaMemcpy(nro_clique_Host, red_test.nro_clique, N * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(tiene_largo_alcance_Host, red_test.tiene_largo_alcance, N * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(vecino_largo_alcance_Host, red_test.vecino_largo_alcance, N * sizeof(int), cudaMemcpyDeviceToHost);

    // cout << endl << "numero de clique : " << endl;
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << nro_clique_Host[i] << ", ";
    // }
    // cout << nro_clique_Host[N-1] << "}\n";

    // cout << endl << "largo alcance : " << endl;
 
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << tiene_largo_alcance_Host[i] << ", ";
    // }
    // cout << tiene_largo_alcance_Host[N-1] << "}\n";


    // cout << endl << "vecinos largo alcance : " << endl;
 
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << vecino_largo_alcance_Host[i] << ", ";
    // }
    // cout << vecino_largo_alcance_Host[N-1] << "}\n";

    // delete [] nro_clique_Host;
    // delete [] tiene_largo_alcance_Host;
    // delete [] vecino_largo_alcance_Host;

    return 0;
}