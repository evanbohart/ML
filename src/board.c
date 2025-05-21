#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "chess.h"

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

board init_board(void)
{
    board b;
    b.white_pieces[KING] = STARTING_WHITE_KING;
    b.white_pieces[PAWN] = STARTING_WHITE_PAWNS;
    b.white_pieces[KNIGHT] = STARTING_WHITE_KNIGHTS;
    b.white_pieces[BISHOP] = STARTING_WHITE_BISHOPS;
    b.white_pieces[ROOK] = STARTING_WHITE_ROOKS;
    b.white_pieces[QUEEN] = STARTING_WHITE_QUEENS;
    b.black_pieces[KING] = STARTING_BLACK_KING;
    b.black_pieces[PAWN] = STARTING_BLACK_PAWNS;
    b.black_pieces[KNIGHT] = STARTING_BLACK_KNIGHTS;
    b.black_pieces[BISHOP] = STARTING_BLACK_BISHOPS;
    b.black_pieces[ROOK] = STARTING_BLACK_ROOKS;
    b.black_pieces[QUEEN] = STARTING_BLACK_QUEENS;

    b.white_pieces_all = b.white_pieces[KING] | b.white_pieces[PAWN] | b.white_pieces[KNIGHT] |
                     b.white_pieces[BISHOP] | b.white_pieces[ROOK] | b.white_pieces[QUEEN];
    b.black_pieces_all = b.black_pieces[KING] | b.black_pieces[PAWN] | b.black_pieces[KNIGHT] |
                     b.black_pieces[BISHOP] | b.black_pieces[ROOK] | b.black_pieces[QUEEN];

    b.pieces_all = b.white_pieces_all | b.black_pieces_all;

    b.white_pins_vertical = 0ULL;
    b.black_pins_vertical = 0ULL;
    b.white_pins_horizontal = 0ULL;
    b.black_pins_horizontal = 0ULL;
    b.white_pins_diagonal1 = 0ULL;
    b.black_pins_diagonal1 = 0ULL;
    b.white_pins_diagonal2 = 0ULL;
    b.black_pins_diagonal2 = 0ULL;

    init_piece_lookup(b.piece_lookup);

    clear_moves(b.legal_moves);
    clear_moves(b.legal_moves);

    return b;
}

void init_piece_lookup(piece *piece_lookup)
{
    piece_lookup[0] = ROOK;
    piece_lookup[1] = KNIGHT;
    piece_lookup[2] = BISHOP;
    piece_lookup[3] = QUEEN;
    piece_lookup[4] = KING;
    piece_lookup[5] = BISHOP;
    piece_lookup[6] = KNIGHT;
    piece_lookup[7] = ROOK;

    for (int i = 8; i < 16; ++i) {
        piece_lookup[i] = PAWN;
    }

    for (int i = 16; i < 48; ++i) {
        piece_lookup[i] = NONE;
    }

    for (int i = 48; i < 56; ++i) {
        piece_lookup[i] = PAWN;
    }

    piece_lookup[56] = ROOK;
    piece_lookup[57] = KNIGHT;
    piece_lookup[58] = BISHOP;
    piece_lookup[59] = QUEEN;
    piece_lookup[60] = KING;
    piece_lookup[61] = BISHOP;
    piece_lookup[62] = KNIGHT;
    piece_lookup[63] = ROOK;
}

void apply_move_white(board *b, move m)
{
    int from = move_from(m);
    int to = move_to(m);
    int flag = move_flag(m);

    if (flag == CAPTURE) {
        piece captured_piece = b->piece_lookup[to];

        clear_bit(b->black_pieces[captured_piece], to);
        clear_bit(b->black_pieces_all, to);
        clear_bit(b->pieces_all, to);
    }

    piece moving_piece = b->piece_lookup[from];

    clear_bit(b->white_pieces[moving_piece], from);
    clear_bit(b->white_pieces_all, from);
    clear_bit(b->pieces_all, from);
    b->piece_lookup[from] = NONE;

    set_bit(b->white_pieces[moving_piece], to);
    set_bit(b->white_pieces_all, to);
    set_bit(b->pieces_all, to);
    b->piece_lookup[to] = moving_piece;
}

void apply_move_black(board *b, move m)
{
    int from = move_from(m);
    int to = move_to(m);
    int flag = move_flag(m);

    if (flag == CAPTURE) {
        piece captured_piece = b->piece_lookup[to];

        clear_bit(b->white_pieces[captured_piece], to);
        clear_bit(b->white_pieces_all, to);
        clear_bit(b->pieces_all, to);
    }

    piece moving_piece = b->piece_lookup[from];

    clear_bit(b->black_pieces[moving_piece], from);
    clear_bit(b->black_pieces_all, from);
    clear_bit(b->pieces_all, from);
    b->piece_lookup[from] = NONE;

    set_bit(b->black_pieces[moving_piece], to);
    set_bit(b->black_pieces_all, to);
    set_bit(b->pieces_all, to);
    b->piece_lookup[to] = moving_piece;
}

void update_board_white(board *b)
{
    get_black_attacks(b);
    get_white_pins(b);
    get_white_moves(b);
}

void update_board_black(board *b)
{
    get_white_attacks(b);
    get_black_pins(b);
    get_black_moves(b);
}

bool white_check(board *b)
{
    return b->white_pieces[KING] & b->black_attacks_all;
}

bool black_check(board *b)
{
    return b->black_pieces[KING] & b->white_attacks_all;
}

bool white_checkmate(board *b)
{
    return white_check(b) && b->legal_moves.count == 0;
}

bool black_checkmate(board *b)
{
    return black_check(b) && b->legal_moves.count == 0;
}

void draw_board(board *b)
{
    char piece_chars[6] = { 'k', 'p', 'n', 'b', 'r', 'q' };

    for (int i = 7; i >= 0; --i) {
        printf("%d ", i + 1);

        for (int j = 0; j < 8; ++j) {
            int pos = i * 8 + j;
            char piece;
            int piece_index = b->piece_lookup[pos];

            if (piece_index == NONE) {
                piece = '.';
            }
            else if (check_bit(b->white_pieces_all, pos)) {
                piece = toupper(piece_chars[piece_index]);
            }
            else {
                piece = piece_chars[piece_index];
            }

            printf("%c ", piece);
        }

        printf("\n");
    }

    printf("  a b c d e f g h\n");
}
