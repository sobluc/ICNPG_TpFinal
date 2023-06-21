/* 
    Header con funciones auxiliares para la implementación de una red de cliques en CPU
*/

#include <ctime>
#include <iostream>
#include <cassert>
using namespace std;


// Ran es un generador de números aleatorios implementado en el libro NUMERICAL RECIPES.
struct Ran {

	unsigned long long int u,v,w;
	Ran(unsigned long long int j) : v(4101842887655102017LL), w(1) {
		u = j ^ v; int64();
		v = u; int64();
		w = v; int64();
	}
	inline unsigned long long int int64() {
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		unsigned long long int x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	inline double doub() { return 5.42101086242752217E-20 * int64(); }
	inline int int32() { return (int)int64(); }
};


Ran ran_gen(time(0)); // Seed para los numeros aleatorios usando RAN de NUMERICAL RECIPES 

// initial_setting_Kronecker genera el numero de clique de cada nodo, establece qué nodos van a tener enlace de largo alcance y 
// asigna el enlace de largo alcance a -1 a todos los nodos. Retorna la cantidad de nodos con enlace de largo alcance.
int initial_setting_Kronecker(int m, float gamma, int N, int* nro_clique, int* tiene_largo_alcance, int* vecino_largo_alcance){

    int clique_count = 0;            
    int count_largo_alcance = 0;    
    
    for(int i = 0; i < N; i++){
        if((i % m == 0) && (i != 0)){ // si alcanzo m nodos cambio el 'nombre' del clique
            clique_count += 1;
        }
        
        nro_clique[i] = clique_count;    
        vecino_largo_alcance[i] = -1;

        float rand_num = ran_gen.doub();
        if (rand_num <= gamma){
            tiene_largo_alcance[i] = 1;
            count_largo_alcance++;
        }
        else{
            tiene_largo_alcance[i] = 0;
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
void asigno_largo_alcance(int N, int cantidad_largo_alcance, int* nro_clique , int* tiene_largo_alcance , int* vecino_largo_alcance){
    // me quedo solo con los nodos que tienen largo alcance
    int* nodos_con_largo_alcance = new int [cantidad_largo_alcance]; 
    assert(nodos_con_largo_alcance != nullptr);

    int aux = 0;
    for(int i = 0; i < N; i++){
        if(tiene_largo_alcance[i] == 1){
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

        // elijo el nodo siguiente y los uno si no estan en el mismo clique
        int nodo1 = nodos_con_largo_alcance[1];

        while((nro_clique[nodo0] == nro_clique[nodo1])){
            shuffle(nodos_con_largo_alcance, cantidad_largo_alcance_reducido);
            nodo0 = nodos_con_largo_alcance[0];
            nodo1 = nodos_con_largo_alcance[1];
        }
        vecino_largo_alcance[nodo0] = nodo1;
        vecino_largo_alcance[nodo1] = nodo0;
        
        cantidad_largo_alcance_reducido -= 2;

        // saco a los nodos nodo0 y nodo1
        int* newArr = new int [cantidad_largo_alcance_reducido];
        for(int i = 2; i < cantidad_largo_alcance_reducido + 2; i++){
            newArr[i - 2] = nodos_con_largo_alcance[i]; 
        }

        delete [] nodos_con_largo_alcance;
        nodos_con_largo_alcance = newArr;
    }

    delete [] nodos_con_largo_alcance;
}