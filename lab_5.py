#!pip install matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


bounds = [(-5, 5), (-5, 5)]


def random_search(func, bounds, n_iter=1000):
    best_solution = None
    best_value = float('inf')
    for _ in range(n_iter):
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        value = func([x, y])
        if value < best_value:
            best_solution = (x, y)
            best_value = value
    return best_solution, best_value


random_solution, random_value = random_search(himmelblau, bounds)


initial_guess = [0, 0]
result = minimize(himmelblau, initial_guess, bounds=bounds)


print("Solucion por busqueda aleatoria")
print(f"  x, y: {random_solution}")
print(f"  Funcion1: {random_value}")

print("\nOptimizandola:")
print(f"  x, y: {result.x}")
print(f"  Funcion2: {result.fun}")


x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
plt.plot(random_solution[0], random_solution[1], 'ro', label='Random Search Solution')
plt.plot(result.x[0], result.x[1], 'bx', label='Optimization Solution')
plt.title('Himmelblau\'s Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.legend()
plt.show()
