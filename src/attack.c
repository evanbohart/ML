#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "chess.h"

const bitboard knight_attack_table[64] = {
    0x0000000000020400, 0x0000000000050800, 0x00000000000a1100, 0x0000000000142200,
    0x0000000000284400, 0x0000000000508800, 0x0000000000a01000, 0x0000000000402000,
    0x0000000002040004, 0x0000000005080008, 0x000000000a110011, 0x0000000014220022,
    0x0000000028440044, 0x0000000050880088, 0x00000000a0100010, 0x0000000040200020,
    0x0000000204000402, 0x0000000508000805, 0x0000000a1100110a, 0x0000001422002214,
    0x0000002844004428, 0x0000005088008850, 0x000000a0100010a0, 0x0000004020002040,
    0x0000020400040200, 0x0000050800080500, 0x00000a1100110a00, 0x0000142200221400,
    0x0000284400442800, 0x0000508800885000, 0x0000a0100010a000, 0x0000402000204000,
    0x0002040004020000, 0x0005080008050000, 0x000a1100110a0000, 0x0014220022140000,
    0x0028440044280000, 0x0050880088500000, 0x00a0100010a00000, 0x0040200020400000,
    0x0204000402000000, 0x0508000805000000, 0x0a1100110a000000, 0x1422002214000000,
    0x2844004428000000, 0x5088008850000000, 0xa0100010a0000000, 0x4020002040000000,
    0x0400040200000000, 0x0800080500000000, 0x1100110a00000000, 0x2200221400000000,
    0x4400442800000000, 0x8800885000000000, 0x100010a000000000, 0x2000204000000000,
    0x0004020000000000, 0x0008050000000000, 0x00110a0000000000, 0x0022140000000000,
    0x0044280000000000, 0x0088500000000000, 0x0010a00000000000, 0x0020400000000000
};

bitboard *bishop_attack_table[64];
bitboard *rook_attack_table[64];

void init_attack_tables(void)
{
    init_bishop_attack_table();
    init_rook_attack_table();
}

void init_bishop_attack_table(void)
{
    for (int i = 0; i < 64; ++i) {
        precompute_bishop_attacks(i);
    }
}

void init_rook_attack_table(void)
{
    for (int i = 0; i < 64; ++i) {
       precompute_rook_attacks(i);
    }
}

bitboard get_blockers(bitboard mask, int x)
{
    bitboard blockers = BITBOARD_ZERO;
    int n = popcount(mask);

    for (int i = 0; i < n; ++i) {
        int bit = ctz(mask);
        clear_bit(mask, bit);

        if (x & (1 << i)) {
            set_bit(blockers, bit);
        }
    }

    return blockers;
}

void precompute_bishop_attacks(int pos)
{
    int n = 1 << (64 - bishop_shifts[pos]);
    bishop_attack_table[pos] = malloc(n * sizeof(bitboard));

    for (int i = 0; i < n; ++i) {
        bitboard blockers = get_blockers(bishop_masks[pos], i);

        bitboard attacks_nw = bishop_rays[pos][0];
        bitboard attacks_ne = bishop_rays[pos][1];
        bitboard attacks_se = bishop_rays[pos][2];
        bitboard attacks_sw = bishop_rays[pos][3];

        int file = pos % 8;
        int rank = pos / 8;

        for (int j = file - 1, k = rank + 1; j >= 0 && k < 8; --j, ++k) {
            if (check_bit(blockers, k * 8 + j)) {
                clear_from(attacks_nw, k * 8 + j);
                break;
            }
        }

        for (int j = file + 1, k = rank + 1; j < 8 && k < 8; ++j, ++k) {
            if (check_bit(blockers, k * 8 + j)) {
                clear_from(attacks_ne, k * 8 + j);
                break;
            }
        }

        for (int j = file + 1, k = rank - 1; j < 8 && k >= 0; ++j, --k) {
            if (check_bit(blockers, k * 8 + j)) {
                clear_until(attacks_se, k * 8 + j);
                break;
            }
        }

        for (int j = file - 1, k = rank - 1; j >= 0 && k >= 0; --j, --k) {
            if (check_bit(blockers, k * 8 + j)) {
                clear_until(attacks_sw, k * 8 + j);
                break;
            }
        }

        int index = bishop_hash(blockers, pos);
        bishop_attack_table[pos][index] = attacks_nw | attacks_ne | attacks_se | attacks_sw;
    }
}

void precompute_rook_attacks(int pos)
{
    int n = 1 << (64 - rook_shifts[pos]);
    rook_attack_table[pos] = malloc(n * sizeof(bitboard));

    for (int i = 0; i < n; ++i) {
        bitboard blockers = get_blockers(rook_masks[pos], i);

        bitboard attacks_up = rook_rays[pos][0];
        bitboard attacks_down = rook_rays[pos][1];
        bitboard attacks_left = rook_rays[pos][2];
        bitboard attacks_right = rook_rays[pos][3];

        int file = pos % 8;
        int rank = pos / 8;

        for (int j = rank + 1; j < 8; ++j) {
            if (check_bit(blockers, j * 8 + file)) {
                clear_from(attacks_up, j * 8 + file);
                break;
            }
        }

        for (int j = rank - 1; j >= 0; --j) {
            if (check_bit(blockers, j * 8 + file)) {
                clear_until(attacks_down, j * 8 + file);
                break;
            }
        }

        for (int j = file - 1; j >= 0; --j) {
            if (check_bit(blockers, rank * 8 + j)) {
                clear_until(attacks_left, rank * 8 + j);
                break;
            }
        }

        for (int j = file + 1; j < 8; ++j) {
            if (check_bit(blockers, rank * 8 + j)) {
                clear_from(attacks_right, rank * 8 + j);
                break;
            }
        }

        int index = rook_hash(blockers, pos);
        rook_attack_table[pos][index] = attacks_up | attacks_down | attacks_left | attacks_right;
    }
}

void destroy_attack_tables(void)
{
    destroy_bishop_attack_table();
    destroy_rook_attack_table();
}

void destroy_bishop_attack_table(void)
{
    for (int i = 0; i < 64; ++i) {
        free(bishop_attack_table[i]);
    }
}

void destroy_rook_attack_table(void)
{
    for (int i = 0; i < 64; ++i) {
        free(rook_attack_table[i]);
    }
}

void get_attacks(board *b, color c)
{
    b->attacks_all[c] = BITBOARD_ZERO;

    get_king_attacks(b, c);
    get_pawn_attacks(b, c);
//    get_knight_attacks(b, c);
    get_bishop_attacks(b, c);
    get_rook_attacks(b, c);
    get_queen_attacks(b, c);
}

void get_king_attacks(board *b, color c)
{
    bitboard king = b->pieces[c][KING];

    b->attacks[c][KING] = BITBOARD_ZERO;

    set_bits(b->attacks[c][KING], calc_shift(king & ~RANK_8, 0, 1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_H & ~RANK_8, 1, 1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_H, 1, 0));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_H & ~RANK_1, 1, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~RANK_1, 0, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_A & ~RANK_1, -1, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_A, -1, 0));
    set_bits(b->attacks[c][KING], calc_shift(king & ~FILE_A & ~RANK_8, -1, 1)); 

    set_bits(b->attacks_all[c], b->attacks[c][KING]);
}

void get_pawn_attacks(board *b, color c)
{
    int dir = c ? -1 : 1;
    bitboard pawns = b->pieces[c][PAWN];

    b->attacks[c][PAWN] = BITBOARD_ZERO;

    set_bits(b->attacks[c][PAWN], calc_shift(pawns & ~FILE_A, -1, dir));
    set_bits(b->attacks[c][PAWN], calc_shift(pawns & ~FILE_H, 1, dir));

    set_bits(b->attacks_all[c], b->attacks[c][PAWN]);
}

void get_knight_attacks(board *b, color c)
{
    bitboard knights = b->pieces[c][KNIGHT];

    b->attacks[c][KNIGHT] = BITBOARD_ZERO;

    while (knights) {
        int pos = ctz(knights);
        clear_bit(knights, pos);

        set_bits(b->attacks[c][KNIGHT], knight_attack_table[pos]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][KNIGHT]);
}

void get_bishop_attacks(board *b, color c)
{
    bitboard bishops = b->pieces[c][BISHOP];

    b->attacks[c][BISHOP] = BITBOARD_ZERO;

    while (bishops) {
        int pos = ctz(bishops);
        clear_bit(bishops, pos);

        int index = bishop_hash(b->pieces_all, pos);

        set_bits(b->attacks[c][BISHOP], bishop_attack_table[pos][index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][BISHOP]);
}

void get_rook_attacks(board *b, color c)
{
    bitboard rooks = b->pieces[c][ROOK];

    b->attacks[c][ROOK] = BITBOARD_ZERO;

    while (rooks) {
        int pos = ctz(rooks);
        clear_bit(rooks, pos);

        int index = rook_hash(b->pieces_all, pos);

        set_bits(b->attacks[c][ROOK], rook_attack_table[pos][index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][ROOK]);
}

void get_queen_attacks(board *b, color c)
{
    bitboard queens = b->pieces[c][QUEEN];

    b->attacks[c][QUEEN] = BITBOARD_ZERO;

    while (queens) {
        int pos = ctz(queens);
        clear_bit(queens, pos);

        int bishop_table_index = bishop_hash(b->pieces_all, pos);
        int rook_table_index = rook_hash(b->pieces_all, pos);

        set_bits(b->attacks[c][QUEEN], bishop_attack_table[pos][bishop_table_index] |
                                       rook_attack_table[pos][rook_table_index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][QUEEN]);
}
