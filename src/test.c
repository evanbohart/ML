#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "chess.h"

bitboard bishop_attacks(bitboard blockers, int pos)
{
    bitboard attacks = 0ULL;

    int file = pos % 8;
    int rank = pos /  8;

    for (int i = file + 1, j = rank + 1; i < 8 && j < 8; ++i, ++j) {
        int sq = j * 8 + i;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    for (int i = file + 1, j = rank - 1; i < 8 && j >= 0; ++i, --j) {
        int sq = j * 8 + i;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    for (int i = file - 1, j = rank - 1; i >= 0 && j >= 0; --i, --j) {
        int sq = j * 8 + i;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    for (int i = file - 1, j = rank + 1; i >= 0 && j < 8; --i, ++j) {
        int sq = j * 8 + i;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    return attacks;
}

bitboard rand_magic(void)
{
    bitboard r = 0;
    for (int i = 0; i < 4; i++) {
        r = (r << 16) | (rand() & 0xFFFF);
    }
    return r;
}

bitboard find_magic(bitboard mask, int relevant_bits, int pos)
{
    int n = 1ULL << relevant_bits;
    bitboard *blockers = malloc(n * sizeof(bitboard));
    bitboard *attack_table = malloc(n * sizeof(bitboard));
    for (int i = 0; i < n; ++i) {
        attack_table[i] = 0ULL;
    }

    for (int i = 0; i < n; ++i) {
        bitboard potential_blockers;
        bool fail = true;
        while (fail) {
            bitboard rand = rand_magic();
            potential_blockers = apply_masks(rand, 1, mask);
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
            bitboard attacks = bishop_attacks(blockers[i], pos);

            if (!(attack_table[(blockers[i] * magic) >> (64 - relevant_bits)])) {
                attack_table[(blockers[i] * magic) >> (64 - relevant_bits)] = attacks;
            }
            else {
                fail = true;
                for (int j = 0; j < n; ++j) {
                    attack_table[j] = 0ULL;
                }
                break;
            }
        }
    }

    free(blockers);
    free(attack_table);

    return magic;
}

int main(void)
{
    srand(time(0));
    /**
    srand(time(0));
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
    printf("WHITE ROOKS\n");
    draw_bitboard(get_rook_attacks(b.white_rooks, b));
    printf("BLACK ROOKS\n");
    draw_bitboard(get_rook_attacks(b.black_rooks, b));
    printf("WHITE QUEEN\n");
    draw_bitboard(get_queen_attacks(b.white_queens, b));
    printf("BLACK QUEEN\n");
    draw_bitboard(get_queen_attacks(b.black_queens, b));
**/

    bitboard magic;

    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 2; ++j) {
            int bits = 64 - bishop_shifts[i * 2 + j];
            magic = find_magic(bishop_masks[i * 2 + j], bits, i * 2 + j);
            printf("%016llxULL, ", magic);
        }
    }

    return 0;
}
