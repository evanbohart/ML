#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "chess.h"

move get_move(void)
{
    char from_file;
    int from_rank;
    char to_file;
    int to_rank;
    int flags;

    printf("from\n");
    scanf(" %c%d", &from_file, &from_rank);
    printf("to\n");
    scanf(" %c%d", &to_file, &to_rank);
    printf("flags\n");
    scanf(" %d", &flags);

    return create_move(from_file - 'a' + 8 * (from_rank - 1), to_file - 'a' + 8 * (to_rank - 1), flags);
}

int main(void)
{
    init_attack_tables();
    board b = init_board();

    while (true) {
        update_board(&b, WHITE);
        bool found = false;
        while (!found) {
            draw_board(&b);
            move white_move = get_move();

            for (int i = 0; i < b.legal_moves.count; ++i) {
                if (white_move == b.legal_moves.moves[i]) {
                    apply_move(&b, WHITE, white_move);
                    found = true;
                    break;
                }
            }

            if (!found) printf("Illegal move.\n");
        }

        update_board(&b, BLACK);
        found = false;
        while (!found) {
            draw_board(&b);
            move black_move = get_move();

            for (int i = 0; i < b.legal_moves.count; ++i) {
                if (black_move == b.legal_moves.moves[i]) {
                    apply_move(&b, BLACK, black_move);
                    found = true;
                    break;
                }
            }

            if (!found) printf("Illegal move.\n");
        }
    }

    destroy_attack_tables();

    return 0;
}
