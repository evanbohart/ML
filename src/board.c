#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "chess.h"

void draw_bitboard(bitboard b)
{
    for (int i = 7; i >= 0; --i) {
        printf("%d ", i + 1);

        for (int j = 0; j < 8; ++j) {
            printf("%llu ", b >> (i * 8 + j) & BITBOARD_ONE);
        }

        printf("\n");
    }

    printf("  a b c d e f g h\n");
}

board init_board(void)
{
    board b;

    b.pieces[WHITE][KING] = STARTING_WHITE_KING;
    b.pieces[WHITE][PAWN] = STARTING_WHITE_PAWNS;
    b.pieces[WHITE][KNIGHT] = STARTING_WHITE_KNIGHTS;
    b.pieces[WHITE][BISHOP] = STARTING_WHITE_BISHOPS;
    b.pieces[WHITE][ROOK] = STARTING_WHITE_ROOKS;
    b.pieces[WHITE][QUEEN] = STARTING_WHITE_QUEENS;
    b.pieces[BLACK][KING] = STARTING_BLACK_KING;
    b.pieces[BLACK][PAWN] = STARTING_BLACK_PAWNS;
    b.pieces[BLACK][KNIGHT] = STARTING_BLACK_KNIGHTS;
    b.pieces[BLACK][BISHOP] = STARTING_BLACK_BISHOPS;
    b.pieces[BLACK][ROOK] = STARTING_BLACK_ROOKS;
    b.pieces[BLACK][QUEEN] = STARTING_BLACK_QUEENS;

    b.pieces_color[WHITE] = b.pieces[WHITE][KING] | b.pieces[WHITE][PAWN] | b.pieces[WHITE][KNIGHT] |
                          b.pieces[WHITE][BISHOP] | b.pieces[WHITE][ROOK] | b.pieces[WHITE][QUEEN];
    b.pieces_color[BLACK] = b.pieces[BLACK][KING] | b.pieces[BLACK][PAWN] | b.pieces[BLACK][KNIGHT] |
                          b.pieces[BLACK][BISHOP] | b.pieces[BLACK][ROOK] | b.pieces[BLACK][QUEEN];

    b.pieces_all = b.pieces_color[WHITE] | b.pieces_color[BLACK];

    b.pins_vertical[WHITE] = BITBOARD_ZERO;
    b.pins_horizontal[WHITE] = BITBOARD_ZERO;
    b.pins_diagonal1[WHITE] = BITBOARD_ZERO;
    b.pins_diagonal2[WHITE] = BITBOARD_ZERO;
    b.pins_vertical[BLACK] = BITBOARD_ZERO;
    b.pins_horizontal[BLACK] = BITBOARD_ZERO;
    b.pins_diagonal1[BLACK] = BITBOARD_ZERO;
    b.pins_diagonal2[BLACK] = BITBOARD_ZERO;

    b.en_passant[WHITE] = BITBOARD_ZERO;
    b.en_passant[BLACK] = BITBOARD_ZERO;

    b.castling_rights[WHITE][0] = true;
    b.castling_rights[WHITE][1] = true;
    b.castling_rights[BLACK][0] = true;
    b.castling_rights[BLACK][1] = true;

    init_piece_lookup(b.piece_lookup);

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

void apply_move(board *b, color c, move m)
{
    int from = move_from(m);
    int to = move_to(m);
    int flag = move_flag(m);

    int king_pos = c ? 60 : 4;
    int short_rook_pos = c ? 63 : 7;
    int long_rook_pos = c ? 56 : 0;

    if (from == king_pos) {
        b->castling_rights[c][0] = false;
        b->castling_rights[c][1] = false;
    }
    else if (from == short_rook_pos) {
        b->castling_rights[c][0] = false;
    }
    else if (from == long_rook_pos) {
        b->castling_rights[c][1] = false;
    }

    b->en_passant[!c] = BITBOARD_ZERO;
    if (flag == DOUBLE_PUSH) {
        int dir = c ? -1 : 1;
        set_bit(b->en_passant[!c], to - 8 * dir);
    }

    if (flag == CASTLE_SHORT) {
        int rook_from = c ? 63 : 7;
        int rook_to = rook_from - 2;

        clear_bit(b->pieces[c][ROOK], rook_from);
        clear_bit(b->pieces_color[c], rook_from);
        clear_bit(b->pieces_all, rook_from);
        b->piece_lookup[rook_from] = NONE;

        set_bit(b->pieces[c][ROOK], rook_to);
        set_bit(b->pieces_color[c], rook_to);
        set_bit(b->pieces_all, rook_to);
        b->piece_lookup[rook_to] = ROOK;
    }

    if (flag == CASTLE_LONG) {
        int rook_from = c ? 56 : 0;
        int rook_to = rook_from + 3;

        clear_bit(b->pieces[c][ROOK], rook_from);
        clear_bit(b->pieces_color[c], rook_from);
        clear_bit(b->pieces_all, rook_from);
        b->piece_lookup[rook_from] = NONE;

        set_bit(b->pieces[c][ROOK], rook_to);
        set_bit(b->pieces_color[c], rook_to);
        set_bit(b->pieces_all, rook_to);
        b->piece_lookup[rook_to] = ROOK;
    }

    if (flag == CAPTURE) {
        piece captured_piece = b->piece_lookup[to];
        clear_bit(b->pieces[!c][captured_piece], to);
        clear_bit(b->pieces_color[!c], to);
        clear_bit(b->pieces_all, to);
    }

    if (flag == EN_PASSANT) {
        int dir = c ? -1 : 1;
        clear_bit(b->pieces[!c][PAWN], to - 8 * dir);
        clear_bit(b->pieces_color[!c], to - 8 * dir);
        clear_bit(b->pieces_all, to - 8 * dir);
        b->piece_lookup[to - 8 * dir] = NONE;
    }

    piece moving_piece = b->piece_lookup[from];

    clear_bit(b->pieces[c][moving_piece], from);
    clear_bit(b->pieces_color[c], from);
    clear_bit(b->pieces_all, from);
    b->piece_lookup[from] = NONE;

    set_bit(b->pieces[c][moving_piece], to);
    set_bit(b->pieces_color[c], to);
    set_bit(b->pieces_all, to);
    b->piece_lookup[to] = moving_piece;
}

void update_board(board *b, color c)
{
    get_attacks(b, c);
    get_attacks(b, !c);
    get_pins(b, c);
    get_legal_moves(b, c);
}

bool check(board *b, color c)
{
    return b->pieces[c][KING] & b->attacks_all[!c];
}

bool checkmate(board *b, color c)
{
    return check(b, c) && b->legal_moves.count == 0;
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
            else if (check_bit(b->pieces_color[WHITE], pos)) {
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
