#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "chess.h"

uint64_t rand_u64(void) {
    return ((uint64_t)rand() << 48) ^ ((uint64_t)rand() << 32) ^
           ((uint64_t)rand() << 16) ^ (uint64_t)rand();
}

bitboard rand_magic(void) {
    bitboard magic;

    do {
        magic = rand_u64() & rand_u64() & rand_u64();
    } while (__builtin_popcountll((magic * 0x9D39247E33776D41ULL) & 0xFF00000000000000ULL) < 6);

    return magic;
}

bitboard find_magic(bitboard mask, int relevant_bits, int pos)
{
    int n = 1ULL << relevant_bits;
    bitboard blockers[n];
    bitboard attack_table[n];
    bool used[n];

    memset(attack_table, 0, sizeof(attack_table));
    memset(used, false, sizeof(used));

    for (int i = 0; i < n; ++i) {
        bitboard potential_blockers;
        bool fail = true;
        while (fail) {
            potential_blockers = apply_masks(rand_u64(), 1, mask);
            fail = false;
            for (int j = 0; j < i; ++j) {
                if (potential_blockers == blockers[j]) {
                    fail = true;
                    break;
                }
            }
        }

        blockers[i] = potential_blockers;
    }

    bitboard magic;
    bool fail = true;

    while (fail) {
        magic = rand_magic();
        fail = false;
        for (int i = 0; i < n; ++i) {
            bitboard attacks = get_bishop_attacks_slow(blockers[i], pos);
            int index = (blockers[i] * magic) >> (64 - relevant_bits);
            if (!used[index]) {
                attack_table[index] = attacks;
                used[index] = true;
            }
            else if (attack_table[index] != attacks) {
                fail = true;
                memset(attack_table, 0, sizeof(attack_table));
                memset(used, false, sizeof(used));
                break;
            }
        }
    }

    return magic;
}

int main(void)
{
    srand(time(0));
    /*init_bishop_attack_table();
    board b = init_board();
    draw_board(b);
    printf("WHITE KING\n");
    draw_bitboard(get_king_attacks(b.white_king));
    printf("BLACK KING\n");
    draw_bitboard(get_king_attacks(b.black_king));
    printf("WHITE PAWNS\n");
    draw_bitboard(get_white_pawn_attacks(b.white_pawns));
    printf("BLACK PAWNS\n");
    draw_bitboard(get_black_pawn_attacks(b.black_pawns));
    printf("WHITE KNIGHTS\n");
    draw_bitboard(get_knight_attacks(b.white_knights));
    printf("BLACK KNIGHTS\n");
    draw_bitboard(get_knight_attacks(b.black_knights));
    printf("WHITE BISHOPS\n");
    draw_bitboard(get_bishop_attacks(b.white_bishops, b));
    printf("BLACK BISHOPS\n");
    draw_bitboard(get_bishop_attacks(b.black_bishops, b));
    //blockersprintf("WHITE ROOKS\n");
    //draw_bitboard(get_rook_attacks(b.white_rooks, b));
    //printf("BLACK ROOKS\n");
    //draw_bitboard(get_rook_attacks(b.black_rooks, b));
    //printf("WHITE QUEEN\n");
    //draw_bitboard(get_queen_attacks(b.white_queens, b));
    //printf("BLACK QUEEN\n");
    //draw_bitboard(get_queen_attacks(b.black_queens, b));
*/
    bitboard magic;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            int bits = 64 - bishop_shifts[i * 4 + j];
            magic = find_magic(bishop_masks[i * 4 + j], bits, i * 4 + j);
            printf("0x%016llxULL, ", magic);
        }
        printf("\n");
    }

    return 0;
}
