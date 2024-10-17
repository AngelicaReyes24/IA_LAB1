import numpy as np
import matplotlib.pyplot as plt

# Definir la función de Himmelblau
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Implementar el algoritmo de Recocido Simulado
def simulated_annealing(func, bounds, initial_temp, final_temp, alpha, max_iter):
    # Generar una solución inicial aleatoria
    current_solution = np.random.uniform(bounds[0], bounds[1], size=2)
    current_value = func(current_solution)
    best_solution = np.copy(current_solution)
    best_value = current_value
    temp = initial_temp

    # Recocido simulado
    for i in range(max_iter):
        # Generar una nueva solución vecina aleatoria
        new_solution = current_solution + np.random.uniform(-0.5, 0.5, size=2)
        # Limitar las nuevas soluciones a los límites de búsqueda
        new_solution = np.clip(new_solution, bounds[0], bounds[1])
        new_value = func(new_solution)

        # Decidir si aceptamos la nueva solución
        if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temp):
            current_solution = new_solution
            current_value = new_value

            # Si es la mejor solución encontrada, actualizarla
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value

        # Reducir la temperatura
        temp *= alpha

        # Detener si se alcanza la temperatura final
        if temp < final_temp:
            break

    return best_solution, best_value

# Parámetros del recocido simulado
bounds = [-5, 5]  # límites de búsqueda
initial_temp = 1000  # temperatura inicial
final_temp = 1e-3  # temperatura final
alpha = 0.9  # factor de enfriamiento
max_iter = 10000  # número máximo de iteraciones

# Encontrar la solución utilizando recocido simulado
solution, value = simulated_annealing(himmelblau, bounds, initial_temp, final_temp, alpha, max_iter)

# Mostrar resultados
print("Mejor solución encontrada (x, y):", solution)
print("Valor de la función en ese punto:", value)

# Gráfico de la función de Himmelblau
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
plt.plot(solution[0], solution[1], 'ro', label='Solución Recocido Simulado')
plt.title('Himmelblau Function with Simulated Annealing Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.legend()
plt.show()
