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

move create_move(int from, int to, int flag)
{
    return (move)from | ((move)to << 6) | ((move)flag << 12);
}

void clear_moves(move_list *l)
{
    l->count = 0;
}

bool add_move(move m, move_list *l)
{
    if (l->count == MAX_MOVES) return false;

    l->moves[l->count++] = m;
    return true;
}

void display_moves(move_list l)
{
    for (int i = 0; i < l.count; ++i) {
        int from = 0x003F & l.moves[i];
        int to = 0x003F & (l.moves[i] >> 6);
        int flags = 0x000F & (l.moves[i] >> 12);

        char from_file = from % 8 + 'a';
        char to_file = to % 8 + 'a';

        int from_rank = from / 8 + 1;
        int to_rank = to / 8 + 1;


        printf("FROM: %c%d\n", from_file, from_rank);
        printf("TO: %c%d\n", to_file, to_rank);
        printf("FLAGS: %d\n", flags);
    }
}

bool get_white_moves(board b, move_list *l)
{
    clear_moves(l);

    if (!get_white_king_moves(b, l) || !get_white_pawn_moves(b, l)
        //!get_white_knight_moves(b, l) || !get_white_bishop_moves(b, l) ||
        //!get_white_rook_moves(b, l) || !get_white_queen_moves(b, l))
    ) return false;

    return true;
}

bool get_black_moves(board b, move_list *l)
{
    clear_moves(l);

    if (!get_black_king_moves(b, l) || !get_black_pawn_moves(b, l)) return false;

    return true;
}

bool get_white_king_moves(board b, move_list *l)
{
    //TODO castling
    bitboard moves = get_king_attacks(b.white_king) & ~get_black_attacks(b) & ~b.white_pieces;

    int from = __builtin_ctzll(b.white_king);
    while (moves) {
        int to = __builtin_ctzll(moves);
        clear_bit(&moves, to);

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    return true;
}

bool get_black_king_moves(board b, move_list *l)
{
    bitboard moves = get_king_attacks(b.black_king) & ~get_white_attacks(b) & ~b.black_pieces;

    int from = __builtin_ctzll(b.black_king);
    while (moves) {
        int to = __builtin_ctzll(moves);
        clear_bit(&moves, to);

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    return true;
}

bool get_white_pawn_moves(board b, move_list *l)
{
    //TODO en passant
    bitboard single_pushes = calc_shift(b.white_pawns, 0, 1) & ~(b.white_pieces | b.black_pieces);
    bitboard double_pushes = apply_masks(calc_shift(single_pushes, 0, 1) &
                                         ~(b.white_pieces | b.black_pieces), 1, ~MASK_4);

    while (single_pushes) {
        int to = __builtin_ctzll(single_pushes);
        clear_bit(&single_pushes, to);
        int from = to - 8;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    while (double_pushes) {
        int to = __builtin_ctzll(double_pushes);
        clear_bit(&double_pushes, to);
        int from = to - 16;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    bitboard captures = get_white_pawn_attacks(b.white_pawns) & b.black_pieces;
    bitboard captures_left = captures & calc_shift(b.white_pawns, -1, 1);
    bitboard captures_right = captures & calc_shift(b.white_pawns, 1, 1);

    while (captures_left) {
        int to = __builtin_ctzll(captures_left);
        clear_bit(&captures_left, to);
        int from = to - 7;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    while (captures_right) {
        int to = __builtin_ctzll(captures_right);
        clear_bit(&captures_right, to);
        int from = to - 9;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    return true;
}

bool get_black_pawn_moves(board b, move_list *l)
{
    bitboard single_pushes = calc_shift(b.black_pawns, 0, -1) & ~(b.white_pieces | b.black_pieces);
    bitboard double_pushes = apply_masks(calc_shift(single_pushes, 0, -1) &
                                         ~(b.white_pieces | b.black_pieces), 1, ~MASK_5);

    while (single_pushes) {
        int to = __builtin_ctzll(single_pushes);
        clear_bit(&single_pushes, to);
        int from = to + 8;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    while (double_pushes) {
        int to = __builtin_ctzll(double_pushes);
        clear_bit(&double_pushes, to);
        int from = to + 16;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    bitboard captures = get_black_pawn_attacks(b.black_pawns) & b.white_pieces;
    bitboard captures_left = captures & calc_shift(b.black_pawns, -1, -1);
    bitboard captures_right = captures & calc_shift(b.black_pawns, 1, -1);

    while (captures_left) {
        int to = __builtin_ctzll(captures_left);
        clear_bit(&captures_left, to);
        int from = to + 9;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    while (captures_right) {
        int to = __builtin_ctzll(captures_right);
        clear_bit(&captures_right, to);
        int from = to + 7;

        if (!add_move(create_move(from, to, 0), l)) return false;
    }

    return true;
}

void get_white_knight_moves(board b, move_list *l)
{
}

void get_black_knight_moves(board b, move_list *l)

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

        int index = bishop_hash(b.white_pieces | b.black_pieces, pos);

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

        int index = rook_hash(b.white_pieces | b.black_pieces, pos);

        set_bits(&attacks, rook_attack_table[pos][index]);
    }

    return attacks;
}

bitboard get_queen_attacks(bitboard queens, board b)
{
    return get_bishop_attacks(queens, b) | get_rook_attacks(queens, b);
}
