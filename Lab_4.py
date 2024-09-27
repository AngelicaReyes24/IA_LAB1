import heapq
import time

# Representación del laberinto
maze = [
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
]

# Coordenadas de inicio y fin
start = (0, 1)  # Coordenadas de inicio
end = (3, 4)  # Coordenadas de fin

# Función para encontrar el camino más corto usando A*
def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    
    # Definición de movimientos posibles (arriba, abajo, izquierda, derecha)
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Función para calcular la distancia heurística (distancia de Manhattan)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Cola de prioridad para los nodos (heapq)
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    # Diccionario para guardar los costos y los padres de cada nodo
    cost_from_start = {start: 0}
    came_from = {start: None}
    
    while open_list:
        # Nodo actual con el costo más bajo
        current_cost, current = heapq.heappop(open_list)
        
        # Si hemos llegado al destino
        if current == end:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Devolvemos el camino en orden correcto (de inicio a fin)
        
        # Explorar los vecinos
        for movement in movements:
            neighbor = (current[0] + movement[0], current[1] + movement[1])
            # Si está dentro de los límites y no es una pared (1)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0:
                new_cost = cost_from_start[current] + 1  # Suponemos costo de 1 por movimiento
                # Si es un nuevo nodo o encontramos un camino más corto
                if neighbor not in cost_from_start or new_cost < cost_from_start[neighbor]:
                    cost_from_start[neighbor] = new_cost
                    priority = new_cost + heuristic(end, neighbor)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current
    return None  # No se encontró un camino

# Medir el tiempo de ejecución
start_time = time.time()
path = astar(maze, start, end)
end_time = time.time()

# Mostrar resultados
if path:
    print(f"Camino encontrado: {path}")
else:
    print("No se encontró un camino.")
    
print(f"Tiempo de ejecución: {end_time - start_time:.6f} segundos")
