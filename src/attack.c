#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "chess.h"

bitboard get_white_attacks(board b)
{
    return get_king_attacks(b.white_king) | get_white_pawn_attacks(b.white_pawns) |
           get_knight_attacks(b.white_knights) | get_bishop_attacks(b.white_bishops, b) |
           get_rook_attacks(b.white_rooks, b) | get_queen_attacks(b.white_queens, b);
}

bitboard get_black_attacks(board b)
{
    return get_king_attacks(b.black_king) | get_black_pawn_attacks(b.black_pawns) |
           get_knight_attacks(b.black_knights) | get_bishop_attacks(b.black_bishops, b) |
           get_rook_attacks(b.black_rooks, b) | get_queen_attacks(b.black_queens, b);
}

bitboard get_king_attacks(bitboard king)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(apply_masks(king, 1, MASK_8), 0, 1));
    set_bits(attacks, calc_shift(apply_masks(king, 2, MASK_H, MASK_8), 1, 1));
    set_bits(attacks, calc_shift(apply_masks(king, 1, MASK_H), 1, 0));
    set_bits(attacks, calc_shift(apply_masks(king, 2, MASK_H, MASK_1), 1, -1));
    set_bits(attacks, calc_shift(apply_masks(king, 1, MASK_1), 0, -1));
    set_bits(attacks, calc_shift(apply_masks(king, 2, MASK_A, MASK_1), -1, -1));
    set_bits(attacks, calc_shift(apply_masks(king, 1, MASK_A), -1, 0));
    set_bits(attacks, calc_shift(apply_masks(king, 2, MASK_A, MASK_8), -1, 1)); 

    return attacks;
}

bitboard get_white_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(apply_masks(pawns, 1, MASK_A), -1, 1));
    set_bits(attacks, calc_shift(apply_masks(pawns, 1, MASK_H), 1, 1));

    return attacks;
}

bitboard get_black_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(apply_masks(pawns, 1, MASK_A), -1, -1));
    set_bits(attacks, calc_shift(apply_masks(pawns, 1, MASK_H), 1, -1));

    return attacks;
}

bitboard get_knight_attacks(bitboard knights)
{
    bitboard attacks = 0ULL;

    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_7, MASK_8), 1, 2));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_G, MASK_8), 2, 1));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_G, MASK_1), 2, -1));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_1, MASK_2), 1, -2));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_1, MASK_2), -1, -2));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_B, MASK_1), -2, -1));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_B, MASK_8), -2, 1));
    set_bits(attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_7, MASK_8), -1, 2));

    return attacks;
}

bitboard get_bishop_attacks(bitboard bishops, board b)
{
    bitboard attacks = 0ULL;

    while (bishops) {
        int pos = __builtin_ctzll(bishops);
        clear_bit(bishops, pos);

        int index = bishop_hash(b.white_pieces | b.black_pieces, pos);

        set_bits(attacks, bishop_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_rook_attacks(bitboard rooks, board b)
{
    bitboard attacks = 0ULL;

    while (rooks) {
        int pos = __builtin_ctzll(rooks);
        clear_bit(rooks, pos);

        int index = rook_hash(b.white_pieces | b.black_pieces, pos);

        set_bits(attacks, rook_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_queen_attacks(bitboard queens, board b)
{
    return get_bishop_attacks(queens, b) | get_rook_attacks(queens, b);
}
