#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_board(char board[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf(" %c ", board[i][j]);
            if (j < 3) printf("|");
        }
        printf("\n");
        if (i < 3) printf("------------\n");
    }
}

int check_winner(char board[4][4], char player) {
    // Check rows and columns
    for (int i = 0; i < 4; i++) {
        if ((board[i][0] == player && board[i][1] == player &&
             board[i][2] == player && board[i][3] == player) ||
            (board[0][i] == player && board[1][i] == player &&
             board[2][i] == player && board[3][i] == player)) {
            return 1;
        }
    }

    // Check diagonals
    if ((board[0][0] == player && board[1][1] == player &&
         board[2][2] == player && board[3][3] == player) ||
        (board[0][3] == player && board[1][2] == player &&
         board[2][1] == player && board[3][0] == player)) {
        return 1;
    }

    return 0;
}

int is_board_full(char board[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == ' ') {
                return 0;
            }
        }
    }
    return 1;
}

void player_move(char board[4][4]) {
    int row, col;
    while (1) {
        printf("INSERTA LA FILA (0-3): ");
        scanf("%d", &row);
        printf("INSERTA LA COLUMNA (0-3): ");
        scanf("%d", &col);
        if (row >= 0 && row < 4 && col >= 0 && col < 4 && board[row][col] == ' ') {
            board[row][col] = 'X';
            break;
        } else {
            printf("MOVIMIENTO INVALIDO.\n");
        }
    }
}

void computer_move(char board[4][4]) {
    int empty_positions[16][2];
    int count = 0;

    // Find all empty positions
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == ' ') {
                empty_positions[count][0] = i;
                empty_positions[count][1] = j;
                count++;
            }
        }
    }

    // Select a random empty position
    if (count > 0) {
        srand(time(NULL));
        int random_index = rand() % count;
        int row = empty_positions[random_index][0];
        int col = empty_positions[random_index][1];
        board[row][col] = 'O';
    }
}

void play_game() {
    char board[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            board[i][j] = ' ';
        }
    }

    printf("GATO 4X4 \n");
    print_board(board);

    while (1) {
        // Player move
        player_move(board);
        print_board(board);
        if (check_winner(board, 'X')) {
            printf("GANASTE\n");
            break;
        }
        if (is_board_full(board)) {
            printf("EMPATE\n");
            break;
        }

        // Computer move
        computer_move(board);
        print_board(board);
        if (check_winner(board, 'O')) {
            printf("PERDISTE\n");
            break;
        }
        if (is_board_full(board)) {
            printf("EMPATE\n");
            break;
        }
    }
}

int main() {
    play_game();
    return 0;
}
