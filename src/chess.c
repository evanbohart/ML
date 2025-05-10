#include <stdio.h>
#include <stdlib.h>
#include "chess.h"

bitboard rand_bitboard(bitboard *occupied, int c)
{
    bitboard b = 0;

    for (int i = 0; i < c; ++i) {
        int pos;
        do {
            pos = rand() % 64;
        }
        while (check_bit(*occupied, pos));

        set_bit(occupied, pos);
        set_bit(&b, pos);
    }

    return b;
}

bitboard rand_king(bitboard *occupied)
{
    return rand_bitboard(occupied, 1);
}

bitboard rand_pawns(bitboard *occupied)
{
    return rand_bitboard(occupied, rand() % 9);
}

bitboard rand_knights(bitboard *occupied)
{
    return rand_bitboard(occupied, rand() % 3);
}

bitboard rand_bishops(bitboard *occupied)
{
    return rand_bitboard(occupied, rand() % 3);
}

bitboard rand_rooks(bitboard *occupied)
{
    return rand_bitboard(occupied, rand() % 3);
}

bitboard rand_queens(bitboard *occupied)
{
    return rand_bitboard(occupied, rand() % 2);
}

void draw_bitboard(bitboard b)
{
    for (int i = 7; i >= 0; --i) {
        printf("%d ", i + 1);

        for (int j = 0; j < 8; ++j) {
            printf("%llu ", (b >> (i * 8 + j)) & 1ULL);
        }

        printf("\n");
    }

    printf("  a b c d e f g h\n");
}

bitboard apply_masks(bitboard b, int count, ...)
{
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; ++i) {
        b &= ~va_arg(args, bitboard);
    }
    va_end(args);

    return b;
}

board init_board(void)
{
    board b;
    b.white_king = STARTING_WHITE_KING;
    b.black_king = STARTING_BLACK_KING;
    b.white_pawns = STARTING_WHITE_PAWNS;
    b.black_pawns = STARTING_BLACK_PAWNS;
    b.white_knights = STARTING_WHITE_KNIGHTS;
    b.black_knights = STARTING_BLACK_KNIGHTS;
    b.white_bishops = STARTING_WHITE_BISHOPS;
    b.black_bishops = STARTING_BLACK_BISHOPS;
    b.white_queens = STARTING_WHITE_QUEENS;
    b.black_queens = STARTING_BLACK_QUEENS;
    b.white_rooks = STARTING_WHITE_ROOKS;
    b.black_rooks = STARTING_BLACK_ROOKS;

    b.white_pieces = b.white_king | b.white_pawns | b.white_knights |
                     b.white_bishops | b.white_rooks | b.white_queens;
    b.black_pieces = b.black_king | b.black_pawns | b.black_knights |
                     b.black_bishops | b.black_rooks | b.black_queens;

    return b;
}

board rand_board(void)
{
    board b;
    bitboard occupied = 0;

    b.white_king = rand_king(&occupied);
    b.black_king = rand_king(&occupied);
    b.white_pawns = rand_pawns(&occupied);
    b.black_pawns = rand_pawns(&occupied);
    b.white_knights = rand_knights(&occupied);
    b.black_knights = rand_knights(&occupied);
    b.white_bishops = rand_bishops(&occupied);
    b.black_bishops = rand_bishops(&occupied);
    b.white_rooks = rand_rooks(&occupied);
    b.black_rooks = rand_rooks(&occupied);
    b.white_queens = rand_queens(&occupied);
    b.black_queens = rand_queens(&occupied);

    b.white_pieces = b.white_king | b.white_pawns | b.white_knights |
                     b.white_bishops | b.white_rooks | b.white_queens;
    b.black_pieces = b.black_king | b.black_pawns | b.black_knights |
                     b.black_bishops | b.black_rooks | b.black_queens;

    return b;
}

void draw_board(board b)
{
    bitboard pieces[12] = { b.white_king, b.white_pawns, b.white_knights,
                            b.white_bishops, b.white_rooks, b.white_queens,
                            b.black_king, b.black_pawns, b.black_knights,
                            b.black_bishops, b.black_rooks, b.black_queens };

    char piece_chars[12] = { 'K', 'P', 'N', 'B', 'R', 'Q',
                             'k', 'p', 'n', 'b', 'r', 'q' };

    for (int i = 7; i >= 0; --i) {
        printf("%d ", i + 1);

        for (int j = 0; j < 8; ++j) {
            char piece = '.';

            for (int k = 0; k < 12; ++k) {
                if (check_bit(pieces[k], i * 8 + j)) {
                    piece = piece_chars[k];
                }
            }

            printf("%c ", piece);
        }

        printf("\n");
    }

    printf("  a b c d e f g h\n");
}

bool white_check(board b)
{
    return b.white_king & get_black_attacks(b);
}

bool black_check(board b)
{
    return b.black_king & get_white_attacks(b);
}

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

    set_bits(&attacks, calc_shift(apply_masks(king, 1, MASK_8), 0, 1));
    set_bits(&attacks, calc_shift(apply_masks(king, 2, MASK_H, MASK_8), 1, 1));
    set_bits(&attacks, calc_shift(apply_masks(king, 1, MASK_H), 1, 0));
    set_bits(&attacks, calc_shift(apply_masks(king, 2, MASK_H, MASK_1), 1, -1));
    set_bits(&attacks, calc_shift(apply_masks(king, 1, MASK_1), 0, -1));
    set_bits(&attacks, calc_shift(apply_masks(king, 2, MASK_A, MASK_1), -1, -1));
    set_bits(&attacks, calc_shift(apply_masks(king, 1, MASK_A), -1, 0));
    set_bits(&attacks, calc_shift(apply_masks(king, 2, MASK_A, MASK_8), -1, 1)); 

    return attacks;
}

bitboard get_white_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(&attacks, calc_shift(apply_masks(pawns, 1, MASK_A), -1, 1));
    set_bits(&attacks, calc_shift(apply_masks(pawns, 1, MASK_H), 1, 1));

    return attacks;
}

bitboard get_black_pawn_attacks(bitboard pawns)
{
    bitboard attacks = 0ULL;

    set_bits(&attacks, calc_shift(apply_masks(pawns, 1, MASK_A), -1, -1));
    set_bits(&attacks, calc_shift(apply_masks(pawns, 1, MASK_H), 1, -1));

    return attacks;
}

bitboard get_knight_attacks(bitboard knights)
{
    bitboard attacks = 0ULL;

    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_7, MASK_8), 1, 2));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_G, MASK_8), 2, 1));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_G, MASK_1), 2, -1));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_H, MASK_1, MASK_2), 1, -2));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_1, MASK_2), -1, -2));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_B, MASK_1), -2, -1));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_B, MASK_8), -2, 1));
    set_bits(&attacks, calc_shift(apply_masks(knights, 3, MASK_A, MASK_7, MASK_8), -1, 2));

    return attacks;
}

bitboard get_bishop_attacks(bitboard bishops, board b)
{
    bitboard attacks = 0ULL;

    while (bishops) {
        int pos = __builtin_ctzll(bishops);
        clear_bit(&bishops, pos);

        bitboard blockers = apply_masks(b.white_pieces | b.black_pieces, 1, bishop_masks[pos]);
        int index = bishop_hash(blockers, pos);

        set_bits(&attacks, bishop_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_rook_attacks(bitboard rooks, board b)
{
    bitboard attacks = 0ULL;

    while (rooks) {
        int pos = __builtin_ctzll(rooks);
        clear_bit(&rooks, pos);

        bitboard blockers = apply_masks(b.white_pieces | b.black_pieces, 1, rook_masks[pos]);
        int index = rook_hash(blockers, pos);

        set_bits(&attacks, rook_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_queen_attacks(bitboard queens, board b)
{
    return get_bishop_attacks(queens, b) | get_rook_attacks(queens, b);
}

bitboard *bishop_attack_table[64];

const bitboard bishop_masks[64] = {
    0x0040201008040200ULL, 0x0000402010080400ULL,
    0x0000004020100a00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204085000ULL,
    0x0000020408102000ULL, 0x0002040810204000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL,
    0x00004020100a0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020408500000ULL,
    0x0002040810200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL,
    0x004020100a000a00ULL, 0x0000402214001400ULL,
    0x0000024428002800ULL, 0x0002040850005000ULL,
    0x0004081020002000ULL, 0x0008102040004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL,
    0x0020100a000a1000ULL, 0x0040221400142200ULL,
    0x0002442800284400ULL, 0x0004085000500800ULL,
    0x0008102000201000ULL, 0x0010204000402000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL,
    0x00100a000a102000ULL, 0x0022140014224000ULL,
    0x0044280028440200ULL, 0x0008500050080400ULL,
    0x0010200020100800ULL, 0x0020400040201000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL,
    0x000a000a10204000ULL, 0x0014001422400000ULL,
    0x0028002844020000ULL, 0x0050005008040200ULL,
    0x0020002010080400ULL, 0x0040004020100800ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL,
    0x00000a1020400000ULL, 0x0000142240000000ULL,
    0x0000284402000000ULL, 0x0000500804020000ULL,
    0x0000201008040200ULL, 0x0000402010080400ULL,
    0x0002040810204000ULL, 0x0004081020400000ULL,
    0x000a102040000000ULL, 0x0014224000000000ULL,
    0x0028440200000000ULL, 0x0050080402000000ULL,
    0x0020100804020000ULL, 0x0040201008040200ULL
};

const bitboard bishop_magic[64];

const int bishop_shifts[64] = { 
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

bitboard *rook_attack_table[64];
const bitboard rook_masks[64];
const bitboard rook_magic[64];
const int rook_shifts[64];

int bishop_hash(bitboard blockers, int pos)
{
    return 0;
}

int rook_hash(bitboard blockers, int pos)
{
    return 0;
}
