#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "chess.h"

void get_attacks(board *b, color c)
{
    b->attacks_all[c]= 0Ull;

    get_king_attacks(b, c);
    get_pawn_attacks(b, c);
    get_knight_attacks(b, c);
    get_bishop_attacks(b, c);
    get_rook_attacks(b, c);
    get_queen_attacks(b, c);
}

void get_king_attacks(board *b, color c)
{
    bitboard king = b->pieces[c][KING];

    b->attacks[c][KING] = 0ULL;

    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_8, 0, 1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_H & ~MASK_8, 1, 1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_H, 1, 0));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_H & ~MASK_1, 1, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_1, 0, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_A & ~MASK_1, -1, -1));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_A, -1, 0));
    set_bits(b->attacks[c][KING], calc_shift(king & ~MASK_A & ~MASK_8, -1, 1)); 

    set_bits(b->attacks_all[c], b->attacks[c][KING]);
}

void get_pawn_attacks(board *b, color c)
{
    int dir = c ? 1 : -1;
    bitboard pawns = b->pieces[c][PAWN];

    b->attacks[c][PAWN] = 0ULL;

    set_bits(b->attacks[c][PAWN], calc_shift(pawns & ~MASK_A, -1, dir));
    set_bits(b->attacks[c][PAWN], calc_shift(pawns & ~MASK_H, 1, dir));

    set_bits(b->attacks_all[c], b->attacks[c][PAWN]);
}

void get_knight_attacks(board *b, color c)
{
    bitboard knights = b->pieces[c][KNIGHT];

    b->attacks[c][KNIGHT] = 0ULL;

    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_H & ~MASK_7 & ~MASK_8, 1, 2));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_H & ~MASK_G & ~MASK_8, 2, 1));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_H & ~MASK_G & ~MASK_1, 2, -1));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_H & ~MASK_1 & ~MASK_2, 1, -2));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_A & ~MASK_1 & ~MASK_2, -1, -2));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_A & ~MASK_B & ~MASK_1, -2, -1));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_A & ~MASK_B & ~MASK_8, -2, 1));
    set_bits(b->attacks[c][KNIGHT], calc_shift(knights & ~MASK_A & ~MASK_7 & ~MASK_8, -1, 2));

    set_bits(b->attacks_all[c], b->attacks[c][KNIGHT]);
}

void get_bishop_attacks(board *b, color c)
{
    bitboard bishops = b->pieces[c][BISHOP];

    b->attacks[c][BISHOP] = 0ULL;

    while (bishops) {
        int pos = ctz(bishops);
        clear_bit(bishops, pos);

        int index = bishop_hash(total_occupancy(*b), pos);

        set_bits(b->attacks[c][BISHOP], bishop_attack_table[pos][index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][BISHOP]);
}

void get_rook_attacks(board *b, color c)
{
    bitboard rooks = b->pieces[c][ROOK];

    b->attacks[c][ROOK] = 0ULL;

    while (rooks) {
        int pos = ctz(rooks);
        clear_bit(rooks, pos);

        int index = rook_hash(total_occupancy(*b), pos);

        set_bits(b->attacks[c][ROOK], rook_attack_table[pos][index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][ROOK]);
}

void get_queen_attacks(board *b, color c)
{
    bitboard queens = b->pieces[c][QUEEN];

    b->attacks[c][QUEEN] = 0ULL;

    while (queens) {
        int pos = ctz(queens);
        clear_bit(queens, pos);

        int bishop_table_index = bishop_hash(total_occupancy(*b), pos);
        int rook_table_index = rook_hash(total_occupancy(*b), pos);

        set_bits(b->attacks[c][QUEEN], bishop_attack_table[pos][bishop_table_index] |
                                       rook_attack_table[pos][rook_table_index]);
    }

    set_bits(b->attacks_all[c], b->attacks[c][QUEEN]);
}
