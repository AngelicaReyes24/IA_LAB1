import math

# Inicializar el tablero 4x4 vacío
def inicializar_tablero():
    return [[" " for _ in range(4)] for _ in range(4)]

# Mostrar el tablero
def mostrar_tablero(tablero):
    for fila in tablero:
        print("|".join(fila))
        print("-" * 7)

# Verificar si hay un ganador
def verificar_ganador(tablero, jugador):
    # Verificar filas, columnas y diagonales
    for i in range(4):
        if all([tablero[i][j] == jugador for j in range(4)]):
            return True
        if all([tablero[j][i] == jugador for j in range(4)]):
            return True
    if all([tablero[i][i] == jugador for i in range(4)]) or all([tablero[i][3-i] == jugador for i in range(4)]):
        return True
    return False

# Evaluar el estado del tablero
def evaluar_tablero(tablero):
    if verificar_ganador(tablero, "X"):
        return 10
    elif verificar_ganador(tablero, "O"):
        return -10
    return 0

# Obtener todos los movimientos posibles
def obtener_movimientos_posibles(tablero):
    movimientos = []
    for i in range(4):
        for j in range(4):
            if tablero[i][j] == " ":
                movimientos.append((i, j))
    return movimientos

# Algoritmo Minimax con poda alfa-beta
def minimax(tablero, profundidad, alfa, beta, es_maximizador):
    puntuacion = evaluar_tablero(tablero)
    
    if puntuacion == 10:
        return puntuacion
    if puntuacion == -10:
        return puntuacion
    if len(obtener_movimientos_posibles(tablero)) == 0:
        return 0
    
    if es_maximizador:
        mejor_valor = -math.inf
        for movimiento in obtener_movimientos_posibles(tablero):
            tablero[movimiento[0]][movimiento[1]] = "X"
            valor = minimax(tablero, profundidad + 1, alfa, beta, False)
            tablero[movimiento[0]][movimiento[1]] = " "
            mejor_valor = max(mejor_valor, valor)
            alfa = max(alfa, mejor_valor)
            if beta <= alfa:
                break
        return mejor_valor
    else:
        mejor_valor = math.inf
        for movimiento in obtener_movimientos_posibles(tablero):
            tablero[movimiento[0]][movimiento[1]] = "O"
            valor = minimax(tablero, profundidad + 1, alfa, beta, True)
            tablero[movimiento[0]][movimiento[1]] = " "
            mejor_valor = min(mejor_valor, valor)
            beta = min(beta, mejor_valor)
            if beta <= alfa:
                break
        return mejor_valor

# Obtener el mejor movimiento para la IA
def obtener_mejor_movimiento(tablero):
    mejor_valor = -math.inf
    mejor_movimiento = None
    for movimiento in obtener_movimientos_posibles(tablero):
        tablero[movimiento[0]][movimiento[1]] = "X"
        movimiento_valor = minimax(tablero, 0, -math.inf, math.inf, False)
        tablero[movimiento[0]][movimiento[1]] = " "
        if movimiento_valor > mejor_valor:
            mejor_movimiento = movimiento
            mejor_valor = movimiento_valor
    return mejor_movimiento

# Manejo del turno del humano
def obtener_movimiento_humano(tablero):
    while True:
        try:
            fila = int(input("Ingresa la fila (0-3): "))
            columna = int(input("Ingresa la columna (0-3): "))
            if tablero[fila][columna] == " ":
                return (fila, columna)
            else:
                print("Movimiento inválido. Inténtalo de nuevo.")
        except:
            print("Entrada inválida. Inténtalo de nuevo.")

# Función para cambiar de turno
def cambiar_turno(jugador):
    return "O" if jugador == "X" else "X"

# Función para el modo Humano vs Humano
def iniciar_humano_vs_humano():
    tablero = inicializar_tablero()
    jugador = "X"
    while True:
        mostrar_tablero(tablero)
        print(f"Turno del jugador {jugador}")
        movimiento = obtener_movimiento_humano(tablero)
        tablero[movimiento[0]][movimiento[1]] = jugador
        if verificar_ganador(tablero, jugador):
            mostrar_tablero(tablero)
            print(f"¡Jugador {jugador} ha ganado!")
            break
        if len(obtener_movimientos_posibles(tablero)) == 0:
            mostrar_tablero(tablero)
            print("¡Es un empate!")
            break
        jugador = cambiar_turno(jugador)

# Función para el modo Humano vs IA
def iniciar_humano_vs_ia():
    tablero = inicializar_tablero()
    jugador = "X"
    while True:
        mostrar_tablero(tablero)
        if jugador == "X":
            print("Tu turno (X):")
            movimiento = obtener_movimiento_humano(tablero)
        else:
            print("Turno de la IA (O):")
            movimiento = obtener_mejor_movimiento(tablero)
        tablero[movimiento[0]][movimiento[1]] = jugador
        if verificar_ganador(tablero, jugador):
            mostrar_tablero(tablero)
            print(f"¡Jugador {jugador} ha ganado!")
            break
        if len(obtener_movimientos_posibles(tablero)) == 0:
            mostrar_tablero(tablero)
            print("¡Es un empate!")
            break
        jugador = cambiar_turno(jugador)

# Función para el modo IA vs IA
def iniciar_ia_vs_ia():
    tablero = inicializar_tablero()
    jugador = "X"
    while True:
        mostrar_tablero(tablero)
        print(f"Turno de la IA ({jugador}):")
        movimiento = obtener_mejor_movimiento(tablero)
        tablero[movimiento[0]][movimiento[1]] = jugador
        if verificar_ganador(tablero, jugador):
            mostrar_tablero(tablero)
            print(f"¡Jugador {jugador} ha ganado!")
            break
        if len(obtener_movimientos_posibles(tablero)) == 0:
            mostrar_tablero(tablero)
            print("¡Es un empate!")
            break
        jugador = cambiar_turno(jugador)

# Menú principal
def main():
    while True:
        print("Selecciona el modo de juego:")
        print("1. Humano vs Humano")
        print("2. Humano vs IA")
        print("3. IA vs IA")
        opcion = input("Elige 1, 2 o 3: ")
        if opcion == "1":
            iniciar_humano_vs_humano()
        elif opcion == "2":
            iniciar_humano_vs_ia()
        elif opcion == "3":
            iniciar_ia_vs_ia()
        else:
            print("Opción inválida. Inténtalo de nuevo.")

# Ejecutar el juego
if __name__ == "__main__":
    main()
