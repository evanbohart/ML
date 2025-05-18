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

        set_bit(*occupied, pos);
        set_bit(b, pos);
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
            printf("%llu ", b >> (i * 8 + j) & 1ULL);
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
        b &= va_arg(args, bitboard);
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

bool white_check(board b)
{
    return b.white_king & get_black_attacks(b);
}

bool black_check(board b)
{
    return b.black_king & get_white_attacks(b);
}

bool white_checkmate(board b)
{
    if (!white_check(b)) return false;

    move_list l;
    get_white_king_moves(b, &l);

    return l.count == 0;
}

bool black_checkmate(board b)
{
    if (black_check(b)) return false;

    move_list l;
    get_black_king_moves(b, &l);

    return l.count == 0;
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
