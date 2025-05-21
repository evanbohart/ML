#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "chess.h"

void get_white_attacks(board *b)
{
    b->white_attacks[KING] = get_king_attacks(b->white_pieces[KING]);
    b->white_attacks[PAWN] = get_black_pawn_attacks(b->white_pieces[PAWN]);
    b->white_attacks[KNIGHT] = get_knight_attacks(b->white_pieces[KNIGHT]);
    b->white_attacks[BISHOP] = get_bishop_attacks(b->white_pieces[BISHOP], b->pieces_all);
    b->white_attacks[ROOK] = get_rook_attacks(b->white_pieces[ROOK], b->pieces_all);
    b->white_attacks[QUEEN] = get_queen_attacks(b->white_pieces[QUEEN], b->pieces_all);

    b->white_attacks_all = b->white_attacks[KING] | b->white_attacks[PAWN] | b->white_attacks[KNIGHT] |
                           b->white_attacks[BISHOP] | b->white_attacks[ROOK] | b->white_attacks[QUEEN];
}

void get_black_attacks(board *b)
{
    b->black_attacks[KING] = get_king_attacks(b->black_pieces[KING]);
    b->black_attacks[PAWN] = get_black_pawn_attacks(b->black_pieces[PAWN]);
    b->black_attacks[KNIGHT] = get_knight_attacks(b->black_pieces[KNIGHT]);
    b->black_attacks[BISHOP] = get_bishop_attacks(b->black_pieces[BISHOP], b->pieces_all);
    b->black_attacks[ROOK] = get_rook_attacks(b->black_pieces[ROOK], b->pieces_all);
    b->black_attacks[QUEEN] = get_queen_attacks(b->black_pieces[QUEEN], b->pieces_all);

    b->black_attacks_all = b->black_attacks[KING] | b->black_attacks[PAWN] | b->black_attacks[KNIGHT] |
                           b->black_attacks[BISHOP] | b->black_attacks[ROOK] | b->black_attacks[QUEEN];
}

bitboard get_king_attacks(bitboard king)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(king & ~MASK_8, 0, 1));
    set_bits(attacks, calc_shift(king & ~MASK_H & ~MASK_8, 1, 1));
    set_bits(attacks, calc_shift(king & ~MASK_H, 1, 0));
    set_bits(attacks, calc_shift(king & ~MASK_H & ~MASK_1, 1, -1));
    set_bits(attacks, calc_shift(king & ~MASK_1, 0, -1));
    set_bits(attacks, calc_shift(king & ~MASK_A & ~MASK_1, -1, -1));
    set_bits(attacks, calc_shift(king & ~MASK_A, -1, 0));
    set_bits(attacks, calc_shift(king & ~MASK_A & ~MASK_8, -1, 1)); 

    return attacks;
}

bitboard get_white_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(pawns & ~MASK_A, -1, 1));
    set_bits(attacks, calc_shift(pawns & ~MASK_H, 1, 1));

    return attacks;
}

bitboard get_black_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(pawns & ~MASK_A, -1, -1));
    set_bits(attacks, calc_shift(pawns & ~MASK_H, 1, -1));

    return attacks;
}

bitboard get_knight_attacks(bitboard knights)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(knights & ~MASK_H & ~MASK_7 & ~MASK_8, 1, 2));
    set_bits(attacks, calc_shift(knights & ~MASK_H & ~MASK_G & ~MASK_8, 2, 1));
    set_bits(attacks, calc_shift(knights & ~MASK_H & ~MASK_G & ~MASK_1, 2, -1));
    set_bits(attacks, calc_shift(knights & ~MASK_H & ~MASK_1 & ~MASK_2, 1, -2));
    set_bits(attacks, calc_shift(knights & ~MASK_A & ~MASK_1 & ~MASK_2, -1, -2));
    set_bits(attacks, calc_shift(knights & ~MASK_A & ~MASK_B & ~MASK_1, -2, -1));
    set_bits(attacks, calc_shift(knights & ~MASK_A & ~MASK_B & ~MASK_8, -2, 1));
    set_bits(attacks, calc_shift(knights & ~MASK_A & ~MASK_7 & ~MASK_8, -1, 2));

    return attacks;
}

bitboard get_bishop_attacks(bitboard bishops, bitboard all_pieces)
{
    bitboard attacks = 0ULL;

    while (bishops) {
        int pos = __builtin_ctzll(bishops);
        clear_bit(bishops, pos);

        int index = bishop_hash(all_pieces, pos);

        set_bits(attacks, bishop_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_rook_attacks(bitboard rooks, bitboard all_pieces)
{
    bitboard attacks = 0ULL;

    while (rooks) {
        int pos = __builtin_ctzll(rooks);
        clear_bit(rooks, pos);

        int index = rook_hash(all_pieces, pos);

        set_bits(attacks, rook_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_queen_attacks(bitboard queens, bitboard all_pieces)
{
    return get_bishop_attacks(queens, all_pieces) | get_rook_attacks(queens, all_pieces);
}
