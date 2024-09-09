#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Definir la función f(x, y)
double f(double x, double y) {
    double term1 = pow((1.5 - x + x * y), 2);
    double term2 = pow((2.25 - x + x * pow(y, 2)), 2);
    double term3 = pow((2.625 - x + x * pow(y, 3)), 2);
    return term1 + term2 + term3;
}

int main() {
    // Definir límites para x e y
    double low = -4.5;
    double high = 4.5;

    // Inicializar variables para el mejor valor encontrado
    double best_x = 0.0, best_y = 0.0;
    double best_value = INFINITY;

    // Inicializar el generador de números aleatorios
    srand(time(0));

    // Bucle de 10,000 iteraciones
    for (int i = 0; i < 10000; i++) {
        // Generar valores aleatorios de x e y dentro del rango [-4.5, 4.5]
        double x = ((double)rand() / RAND_MAX) * (high - low) + low;
        double y = ((double)rand() / RAND_MAX) * (high - low) + low;

        // Calcular el valor de la función en (x, y)
        double value = f(x, y);

        // Si el valor es menor que el mejor valor actual, actualizar
        if (value < best_value) {
            best_value = value;
            best_x = x;
            best_y = y;
        }
    }

    // Imprimir el mejor valor encontrado y los correspondientes valores de x e y
    printf("Mejor valor encontrado:\n");
    printf("x = %f, y = %f\n", best_x, best_y);
    printf("Valor de f(x, y) = %f\n", best_value);

    return 0;
}
