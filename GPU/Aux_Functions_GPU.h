/* 
    Header con funciones auxiliares para la implementación de una red de cliques en CPU
*/

#include <ctime>
#include <iostream>
#include <cassert>
using namespace std;


// Uso el ejemplo de MiniIsing para generar numeros aleatorios

#include <Random123/philox.h> 
#include <Random123/u01.h>    
typedef r123::Philox2x32 RNG; 

int globalseed=123456; // seed global del generador de números aleatorios en paralelo

__device__
float uniform(int n, int seed, int t)
{ 
		RNG philox; 	
		RNG::ctr_type c={{}};
		RNG::key_type k={{}};
		RNG::ctr_type r;

		k[0]=n;    
		c[1]=seed; // seed global, necesario para decidir reproducir secuencia, o no...
		c[0]=t;    // el seed tiene que cambiar con la iteracion, sino...
		r = philox(c, k); // son dos numeros random, usaremos uno solo r[0]
		return (u01_closed_closed_32_53(r[0])); // funcion adaptadora a [0,1]
}

__global__ 
void node_initial_setting(int N, float gamma, int* nro_clique, bool* tiene_largo_alcance){
    int i = blockIdx.x * blockDim.x + threadId.x;
    if(i < N){
        nro_clique[i] = blockIdx;
        float rand_num = uniform(n, seed, t); // misma funcion que en el ejemplo MiniIsing
        tiene_largo_alcance[i] = uniform(n, seed, t) <= gamma;    
    
    }
}


// initial_setting_Kronecker genera el numero de clique de cada nodo, establece qué nodos van a tener enlace de largo alcance y 
// asigna el enlace de largo alcance a -1 a todos los nodos. Retorna la cantidad de nodos con enlace de largo alcance.
int initial_setting_Kronecker(int m, float gamma, int N, int* nro_clique, bool* tiene_largo_alcance, int* vecino_largo_alcance){
    
    // llamo a kernel con Q = N/m bloques y m threads por bloque
    node_initial_setting<<< N/m , m >>>(nro_clique);


    int count_largo_alcance = 0; // calcular la suma con thrust o alguna libreria    
    for(int i = 0; i < N; i++){
            
        vecino_largo_alcance[i] = -1;

        float rand_num = ran_gen.doub();
        if (rand_num <= gamma){
            tiene_largo_alcance[i] = true;
            count_largo_alcance++;
        }
        else{
            tiene_largo_alcance[i] = false;
        }
    }

    return count_largo_alcance;
}





// swap y shuffle son funciones que implementan el algoritmo de Fisher-Yates (https://es.wikipedia.org/wiki/Algoritmo_de_Fisher-Yates)
void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle (int* arr, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = ran_gen.int64() % (i + 1);
        swap(&arr[i], &arr[j]);
    }
}

// asigno_largo_alcance asigna entre cuáles nodos va a existir un enlace de largo alcance
void asigno_largo_alcance(int N, int cantidad_largo_alcance, int* nro_clique , bool* tiene_largo_alcance , int* vecino_largo_alcance){
    // me quedo solo con los nodos que tienen largo alcance
    int* nodos_con_largo_alcance = new int [cantidad_largo_alcance]; 
    assert(nodos_con_largo_alcance != nullptr);

    int aux = 0;
    for(int i = 0; i < N; i++){
        if(tiene_largo_alcance[i]){
            nodos_con_largo_alcance[aux] = i;
            aux++;
        }
    }
    
    int cantidad_largo_alcance_reducido = cantidad_largo_alcance;

    // mezclo la lista
    shuffle(nodos_con_largo_alcance, cantidad_largo_alcance_reducido); 
    
    // uno nodos en diferentes cliques y los remuevo de la lista 
    while(cantidad_largo_alcance_reducido > 0){
        
        int nodo0 = nodos_con_largo_alcance[0];

        // me fijo que los nodos que quedan esten en cliques distintos
        int clique0_num = nro_clique[nodo0];
        int clique_count = nro_clique[nodo0];
        int counter = 0;
        while(clique_count == clique0_num){
            
            if(counter == cantidad_largo_alcance_reducido){ break; }

            clique_count = nro_clique[counter];   
            counter++;
        }

        if(counter >= cantidad_largo_alcance_reducido){ 
            for(int i = 0; i < cantidad_largo_alcance_reducido; i++){
                int aux_nodo = nodos_con_largo_alcance[i];
                tiene_largo_alcance[aux_nodo] = false;
                vecino_largo_alcance[aux_nodo] = -1; 
            }
            break; 
        }

        int nodo1 = nodos_con_largo_alcance[1];

        while((nro_clique[nodo0] == nro_clique[nodo1])){
            shuffle(nodos_con_largo_alcance, cantidad_largo_alcance_reducido);
            nodo0 = nodos_con_largo_alcance[0];
            nodo1 = nodos_con_largo_alcance[1];
        }

        vecino_largo_alcance[nodo0] = nodo1;
        vecino_largo_alcance[nodo1] = nodo0;

        cantidad_largo_alcance_reducido -= 2;

        // saco a los nodos de la lista
        int* newArr = new int [cantidad_largo_alcance_reducido];
        for(int i = 2; i < cantidad_largo_alcance_reducido + 2; i++){
            newArr[i - 2] = nodos_con_largo_alcance[i]; 
        }

        delete [] nodos_con_largo_alcance;
        nodos_con_largo_alcance = newArr;
    }

    delete [] nodos_con_largo_alcance;
}