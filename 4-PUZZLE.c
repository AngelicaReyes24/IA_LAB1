#include <stdio.h>
#include <stdlib.h>
#define FILAS 6
#define COLUMNAS 5
#define MAX 100

// Librería Pila
typedef struct {
    int elementos[MAX];
    int tope;
} Pila;

void inicializar_pila(Pila* p) {
    p->tope = -1;
}

int esta_vacia_pila(Pila* p) {
    return p->tope == -1;
}

int esta_llena_pila(Pila* p) {
    return p->tope == MAX - 1;
}

void push(Pila* p, int valor) {
    if (!esta_llena_pila(p)) {
        p->tope++;
        p->elementos[p->tope] = valor;
    } else {
        printf("Pila llena\n");
    }
}

int pop(Pila* p) {
    if (!esta_vacia_pila(p)) {
        int valor = p->elementos[p->tope];
        p->tope--;
        return valor;
    }
    printf("Pila vacía\n");
    return -1;
}

// Librería Lista Genérica
typedef struct {
    int elementos[MAX];
    int tamano;
} ListaGenerica;

void inicializar_lista(ListaGenerica* l) {
    l->tamano = 0;
}

void insertar_lista(ListaGenerica* l, int valor) {
    if (l->tamano < MAX) {
        l->elementos[l->tamano] = valor;
        l->tamano++;
    } else {
        printf("Lista llena\n");
    }
}

int buscar_lista(ListaGenerica* l, int valor) {
    for (int i = 0; i < l->tamano; i++) {
        if (l->elementos[i] == valor) {
            return i;
        }
    }
    return -1;
}

// Librería Cola
typedef struct {
    int elementos[MAX];
    int frente, final;
} Cola;

void inicializar_cola(Cola* c) {
    c->frente = 0;
    c->final = -1;
}

int esta_vacia_cola(Cola* c) {
    return c->final < c->frente;
}

int esta_llena_cola(Cola* c) {
    return c->final == MAX - 1;
}

void insertar(Cola* c, int valor) {
    if (!esta_llena_cola(c)) {
        c->final++;
        c->elementos[c->final] = valor;
    } else {
        printf("Cola llena\n");
    }
}

int quitar(Cola* c) {
    if (!esta_vacia_cola(c)) {
        int valor = c->elementos[c->frente];
        c->frente++;
        return valor;
    }
    printf("Cola vacía\n");
    return -1;
}

// Laberinto proporcionado en la imagen
int laberinto[FILAS][COLUMNAS] = {
    {1, 0, 1, 1, 1},
    {1, 0, 0, 0, 1},
    {1, 1, 1, 0, 1},
    {1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1}
};

// Coordenadas de inicio y final
int inicio_x = 0, inicio_y = 1;
int fin_x = 3, fin_y = 4;

// Verificar si una posición es válida
int es_valido(int x, int y, int visitados[FILAS][COLUMNAS]) {
    return (x >= 0 && x < FILAS && y >= 0 && y < COLUMNAS && laberinto[x][y] == 0 && !visitados[x][y]);
}

// Movimientos posibles: arriba, abajo, izquierda, derecha
int movimientos[4][2] = {
    {-1, 0}, // arriba
    {1, 0},  // abajo
    {0, -1}, // izquierda
    {0, 1}   // derecha
};

// Implementación de BFS para resolver el laberinto utilizando Cola
void bfs_laberinto() {
    Cola cola;
    inicializar_cola(&cola);
    
    // Matriz de nodos visitados
    int visitados[FILAS][COLUMNAS] = {0};
    
    // Inicializar con la posición de inicio
    insertar(&cola, inicio_x * COLUMNAS + inicio_y);
    visitados[inicio_x][inicio_y] = 1;
    
    while (!esta_vacia_cola(&cola)) {
        int actual = quitar(&cola);
        int x = actual / COLUMNAS;
        int y = actual % COLUMNAS;
        
        // Si encontramos el final
        if (x == fin_x && y == fin_y) {
            printf("¡Camino encontrado! Llegamos a la posición (%d, %d)\n", x, y);
            return;
        }
        
        // Explorar las direcciones posibles
        for (int i = 0; i < 4; i++) {
            int nuevo_x = x + movimientos[i][0];
            int nuevo_y = y + movimientos[i][1];
            
            if (es_valido(nuevo_x, nuevo_y, visitados)) {
                insertar(&cola, nuevo_x * COLUMNAS + nuevo_y);
                visitados[nuevo_x][nuevo_y] = 1;
            }
        }
    }
    
    printf("No se encontró un camino en el laberinto\n");
}

int main() {
    bfs_laberinto();
    return 0;
}
