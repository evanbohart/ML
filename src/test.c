#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "chess.h"

uint64_t rand_u64(void) {
    return (uint64_t)rand() << 48 | (uint64_t)rand() << 32 |
           (uint64_t)rand() << 16 | (uint64_t)rand();
}

bitboard rand_magic(void) {
    return rand_u64() & rand_u64() & rand_u64();
}

bitboard find_magic(bitboard mask, int relevant_bits, int pos)
{
    int n = 1 << relevant_bits;
    bitboard blockers[n];
    bitboard attack_table[n];
    bool used[n];

    memset(attack_table, 0, sizeof(attack_table));
    memset(used, 0, sizeof(used));

    for (int i = 0; i < n; ++i) {
        blockers[i] = get_blockers(mask, i);
    }

    bitboard magic;
    bool fail = true;
    while (fail) {
        magic = rand_magic();
        fail = false;
        for (int i = 0; i < n; ++i) {
            bitboard attacks = get_rook_attacks_slow(blockers[i], pos);
            int index = (blockers[i] * magic) >> (64 - relevant_bits);
            if (!used[index]) {
                attack_table[index] = attacks;
                used[index] = true;
            }
            else if (attack_table[index] != attacks) {
                fail = true;
                memset(attack_table, 0, sizeof(attack_table));
                memset(used, 0, sizeof(used));
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
    init_rook_attack_table();
    board b = rand_board();
    draw_board(b);

    move_list white_moves;
    white_moves.count = 0;
    move_list black_moves;
    black_moves.count = 0;
    get_white_moves(b, &white_moves);
    get_black_moves(b, &black_moves);
    display_moves(white_moves);
    printf("---------------------------\n");
    display_moves(black_moves);*/

    /*
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
    printf("WHITE ROOKS\n");
    draw_bitboard(get_rook_attacks(b.white_rooks, b));
    printf("BLACK ROOKS\n");
    draw_bitboard(get_rook_attacks(b.black_rooks, b));
    printf("WHITE QUEEN\n");
    draw_bitboard(get_queen_attacks(b.white_queens, b));
    printf("BLACK QUEEN\n");
    draw_bitboard(get_queen_attacks(b.black_queens, b));
/*
    bitboard magic;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            int bits = 64 - rook_shifts[i * 4 + j];
            magic = find_magic(rook_masks[i * 4 + j], bits, i * 4 + j);
            printf("0x%016llxULL, ", magic);
        }
        printf("\n");
    }

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            bitboard mask = get_rook_attacks_slow(0ULL, i * 4 + j);

            if (check_bit(MASK_1, i * 4 + j)) {
                mask = apply_masks(mask, 1, MASK_1);
            }
            if (check_bit(MASK_8, i * 4 + j)) {
                mask = apply_masks(mask, 1, MASK_8);
            }
            if (check_bit(MASK_A, i * 4 + j)) {
                mask = apply_masks(mask, 1, MASK_A);
            }
            if (check_bit(MASK_H, i * 4 + j)) {
                mask = apply_masks(mask, 1, MASK_H);
            }

            printf("0x%016llxULL, ", mask);
        }
        printf("\n");
    }*/

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("0x%016llxULL, ", get_bishop_attacks_slow(0ULL, i * 4 + j));
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("0x%016llxULL, ", get_rook_attacks_slow(0ULL, i * 4 + j));
        }
        printf("\n");
    }

    return 0;
}
