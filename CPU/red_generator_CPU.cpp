#include "Aux_Functions_CPU.h"
#include "cpu_timer.h"

class Red_CPU{
    public:
        Red_CPU(int Q, float gamma, int m){ // constructor para cliques iguales
            this -> Q = Q; // cantidad de cliques
            this -> gamma = gamma; // probabilidad de enlaces de largo alcance
            this -> n_max = m; // tamaño de los cliques (en este caso todos constantes)
            this -> N = Q * m; // cantidad de nodos en la red
            this -> clique_sizes = nullptr; // va a servir cuando los cliques cambien de tamaño
            this -> nro_clique = new int [N]; // array con numero de clique
            this -> tiene_largo_alcance = new int [N]; // array con 0 o 1
            this -> vecino_largo_alcance = new int [N]; // array con el vecino de cada nodo

            // establezco los nodos que van a tener enlace de largo alcance y escribo el numero de clique de cada nodo (ver "Aux_Functions_CPU.h")
            int count_largo_alcance = initial_setting_Kronecker(m, gamma, N, nro_clique, tiene_largo_alcance, vecino_largo_alcance); 

            // asigno vecinos de largo alcance (ver "Aux_Functions_CPU.h")
            asigno_largo_alcance(N, count_largo_alcance, nro_clique, tiene_largo_alcance, vecino_largo_alcance);

        }

        ~Red_CPU(){
            if (clique_sizes != nullptr){   
                delete [] clique_sizes;
            }
            if (nro_clique != nullptr){
                delete [] nro_clique;
            }
            if (tiene_largo_alcance != nullptr){
                delete [] tiene_largo_alcance;
            }
            if (vecino_largo_alcance != nullptr){
                delete [] vecino_largo_alcance;
            }
        }


        int Q; // cantidad de cliques
        int n_max; // maximo tamaño que puede tener un clique
        int N;
        float gamma;
        int* clique_sizes; // por ahora va a ser nullptr porque trabajo con tamaño constante
        int* nro_clique; 
        int* tiene_largo_alcance;
        int* vecino_largo_alcance;
};


int main(){

    // Asigno valores de Q (cantidad de cliques), m (tamaño de los cliques) y gamma (probabilidad de largo alcance)
    int Q = 10;
    int m = 3;
    float gamma = 1.0;

    cout << "Q : " << Q << "| m : " << m << " | gamma : " << gamma << endl;

    cpu_timer relojcpu;

    relojcpu.tic();
    Red_CPU red_test (Q, gamma, m); // creo la red con su constructor
    cout << relojcpu.tac() << endl;

    // Las siguientes lineas comentadas son para ver los vectores de la red creada (se recomienda usar con redes chicas)

    // int N = m*Q;

    // cout << endl << "numero de clique : " << endl;
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << red_test.nro_clique[i] << ", ";
    // }
    // cout << red_test.nro_clique[N-1] << "}\n";

    // cout << endl << "largo alcance : " << endl;
 
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << red_test.tiene_largo_alcance[i] << ", ";
    // }
    // cout << red_test.tiene_largo_alcance[N-1] << "}\n";


    // cout << endl << "vecinos largo alcance : " << endl;
 
    // cout << "{";
    // for (size_t i = 0; i < N-1; ++i)
    // {
    //     cout << red_test.vecino_largo_alcance[i] << ", ";
    // }
    // cout << red_test.vecino_largo_alcance[N-1] << "}\n";


    return 0;
}