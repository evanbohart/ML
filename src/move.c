#include <stdio.h>
#include <assert.h>
#include "chess.h"

void clear_moves(move_list *l)
{
    l->count = 0;
}

void add_move(move m, move_list *l)
{
    assert(l->count != MAX_MOVES);

    l->moves[l->count++] = m;
}

void display_moves(move_list l)
{
    for (int i = 0; i < l.count; ++i) {
        int from = move_from(l.moves[i]);
        int to = move_to(l.moves[i]);
        int flag = move_flag(l.moves[i]);

        char from_file = from % 8 + 'a';
        char to_file = to % 8 + 'a';

        int from_rank = from / 8 + 1;
        int to_rank = to / 8 + 1;


        printf("FROM: %c%d\n", from_file, from_rank);
        printf("TO: %c%d\n", to_file, to_rank);
        printf("FLAG: %d\n", flag);
        printf("------------\n");
    }
}

void get_white_moves(board b, move_list *l)
{
    clear_moves(l);

    get_white_king_moves(b, l);
    get_white_pawn_moves(b, l);
    get_white_knight_moves(b, l);
    get_white_bishop_moves(b, l);
    get_white_rook_moves(b, l);
    get_white_queen_moves(b, l);
}

void get_black_moves(board b, move_list *l)
{
    clear_moves(l);

    get_black_king_moves(b, l);
    get_black_pawn_moves(b, l);
    get_black_knight_moves(b, l);
    get_black_bishop_moves(b, l);
    get_black_rook_moves(b, l);
    get_black_queen_moves(b, l);
}

void get_white_king_moves(board b, move_list *l)
{
    bitboard attacks = get_king_attacks(b.white_king);
    bitboard non_captures = attacks & ~get_black_attacks(b) & ~(b.white_pieces | b.black_pieces);
    bitboard captures = attacks & ~get_black_attacks(b) & b.black_pieces;

    int from = __builtin_ctzll(b.white_king);

    while (non_captures) {
        int to = __builtin_ctzll(non_captures);
        clear_bit(non_captures, to);

        add_move(create_move(from, to, QUIET), l);
    }

    while (captures) {
        int to = __builtin_ctzll(captures);
        clear_bit(captures, to);

        add_move(create_move(from, to, CAPTURE), l);
    }
}

void get_black_king_moves(board b, move_list *l)
{
    bitboard attacks = get_king_attacks(b.black_king);
    bitboard non_captures = attacks & ~get_white_attacks(b) & ~(b.white_pieces | b.black_pieces);
    bitboard captures = attacks & ~get_white_attacks(b) & b.white_pieces;

    int from = __builtin_ctzll(b.black_king);

    while (non_captures) {
        int to = __builtin_ctzll(non_captures);
        clear_bit(non_captures, to);

        add_move(create_move(from, to, QUIET), l);
    }

    while (captures) {
        int to = __builtin_ctzll(captures);
        clear_bit(captures, to);

        add_move(create_move(from, to, CAPTURE), l);
    }
}

void get_white_pawn_moves(board b, move_list *l)
{
    bitboard single_pushes = calc_shift(b.white_pawns, 0, 1) & ~(b.white_pieces | b.black_pieces);
    bitboard double_pushes = apply_masks(calc_shift(single_pushes, 0, 1) &
                                         ~(b.white_pieces | b.black_pieces), 1, ~MASK_4);
    bitboard promotions = apply_masks(single_pushes, 1, ~MASK_8);
    bitboard quiet_pushes = apply_masks(single_pushes, 1, MASK_8);

    while (quiet_pushes) {
        int to = __builtin_ctzll(quiet_pushes);
        clear_bit(quiet_pushes, to);
        int from = to - 8;

        add_move(create_move(from, to, QUIET), l);
    }

    while (double_pushes) {
        int to = __builtin_ctzll(double_pushes);
        clear_bit(double_pushes, to);
        int from = to - 16;

        add_move(create_move(from, to, DOUBLE_PUSH), l);
    }

    while (promotions) {
        int to = __builtin_ctzll(promotions);
        clear_bit(promotions, to);
        int from = to - 8;

        add_move(create_move(from, to, PROMO_KNIGHT), l);
        add_move(create_move(from, to, PROMO_BISHOP), l);
        add_move(create_move(from, to, PROMO_ROOK), l);
        add_move(create_move(from, to, PROMO_QUEEN), l);
    }

    bitboard captures = get_white_pawn_attacks(b.white_pawns) & b.black_pieces;
    bitboard captures_left = captures & calc_shift(b.white_pawns, -1, 1);
    bitboard captures_right = captures & calc_shift(b.white_pawns, 1, 1);
    bitboard promotions_left = apply_masks(captures_left, 1, MASK_8);
    bitboard promotions_right = apply_masks(captures_right, 1, MASK_8);
    captures_left = apply_masks(captures_left, 1, ~MASK_8);
    captures_right = apply_masks(captures_right, 1, ~MASK_8);

    while (captures_left) {
        int to = __builtin_ctzll(captures_left);
        clear_bit(captures_left, to);
        int from = to - 7;

        add_move(create_move(from, to, CAPTURE), l);
    }

    while (captures_right) {
        int to = __builtin_ctzll(captures_right);
        clear_bit(captures_right, to);
        int from = to - 9;

        add_move(create_move(from, to, CAPTURE), l);
    }

    while (promotions_left) {
        int to = __builtin_ctzll(promotions_left);
        clear_bit(promotions_left, to);
        int from = to - 7;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), l);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), l);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), l);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), l);
    }

    while (promotions_right) {
        int to = __builtin_ctzll(promotions_right);
        clear_bit(promotions_right, to);
        int from = to - 9;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), l);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), l);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), l);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), l);
    }
}

void get_black_pawn_moves(board b, move_list *l)
{
    bitboard single_pushes = calc_shift(b.black_pawns, 0, -1) & ~(b.white_pieces | b.black_pieces);
    bitboard double_pushes = apply_masks(calc_shift(single_pushes, 0, -1) &
                                         ~(b.white_pieces | b.black_pieces), 1, ~MASK_5);
    bitboard promotions = apply_masks(single_pushes, 1, ~MASK_1);
    bitboard quiet_pushes = apply_masks(single_pushes, 1, MASK_1);

    while (quiet_pushes) {
        int to = __builtin_ctzll(quiet_pushes);
        clear_bit(quiet_pushes, to);
        int from = to + 8;

        add_move(create_move(from, to, QUIET), l);
    }

    while (double_pushes) {
        int to = __builtin_ctzll(double_pushes);
        clear_bit(double_pushes, to);
        int from = to + 16;

        add_move(create_move(from, to, DOUBLE_PUSH), l);
    }

    while (promotions) {
        int to = __builtin_ctzll(promotions);
        clear_bit(promotions, to);
        int from = to + 8;

        add_move(create_move(from, to, PROMO_KNIGHT), l);
        add_move(create_move(from, to, PROMO_BISHOP), l);
        add_move(create_move(from, to, PROMO_ROOK), l);
        add_move(create_move(from, to, PROMO_QUEEN), l);
    }

    bitboard captures = get_black_pawn_attacks(b.black_pawns) & b.white_pieces;
    bitboard captures_left = captures & calc_shift(b.black_pawns, -1, -1);
    bitboard captures_right = captures & calc_shift(b.black_pawns, 1, -1);
    bitboard promotions_left = apply_masks(captures_left, 1, ~MASK_1);
    bitboard promotions_right = apply_masks(captures_right, 1, ~MASK_1);
    captures_left = apply_masks(captures_left, 1, MASK_1);
    captures_right = apply_masks(captures_right, 1, MASK_1);


    while (captures_left) {
        int to = __builtin_ctzll(captures_left);
        clear_bit(captures_left, to);
        int from = to + 9;

        add_move(create_move(from, to, CAPTURE), l);
    }

    while (captures_right) {
        int to = __builtin_ctzll(captures_right);
        clear_bit(captures_right, to);
        int from = to + 7;

        add_move(create_move(from, to, CAPTURE), l);
    }

    while (promotions_left) {
        int to = __builtin_ctzll(promotions_left);
        clear_bit(promotions_left, to);
        int from = to + 9;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), l);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), l);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), l);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), l);
    }

    while (promotions_right) {
        int to = __builtin_ctzll(promotions_right);
        clear_bit(promotions_right, to);
        int from = to + 7;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), l);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), l);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), l);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), l);
    }
}

void get_white_knight_moves(board b, move_list *l)
{
    bitboard knights = b.white_knights;

    while (knights) {
        int from = __builtin_ctzll(knights);
        clear_bit(knights, from);

        bitboard attacks = get_knight_attacks(1ULL << from);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.black_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_black_knight_moves(board b, move_list *l)
{
    bitboard knights = b.black_knights;

    while (knights) {
        int from = __builtin_ctzll(knights);
        clear_bit(knights, from);

        bitboard attacks = get_knight_attacks(1ULL << from);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.white_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_white_bishop_moves(board b, move_list *l)
{
    bitboard bishops = b.white_bishops;

    while (bishops) {
        int from = __builtin_ctzll(bishops);
        clear_bit(bishops, from);

        bitboard attacks = get_bishop_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.black_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_black_bishop_moves(board b, move_list *l)
{
    bitboard bishops = b.black_bishops;

    while (bishops) {
        int from = __builtin_ctzll(bishops);
        clear_bit(bishops, from);

        bitboard attacks = get_bishop_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.white_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_white_rook_moves(board b, move_list *l)
{
    bitboard rooks = b.white_rooks;

    while (rooks) {
        int from = __builtin_ctzll(rooks);
        clear_bit(rooks, from);

        bitboard attacks = get_rook_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.black_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_black_rook_moves(board b, move_list *l)
{
    bitboard rooks = b.black_rooks;

    while (rooks) {
        int from = __builtin_ctzll(rooks);
        clear_bit(rooks, from);

        bitboard attacks = get_rook_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.white_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_white_queen_moves(board b, move_list *l)
{
    bitboard queens = b.white_queens;

    while (queens) {
        int from = __builtin_ctzll(queens);
        clear_bit(queens, from);

        bitboard attacks = get_queen_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.black_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}

void get_black_queen_moves(board b, move_list *l)
{
    bitboard queens = b.black_queens;

    while (queens) {
        int from = __builtin_ctzll(queens);
        clear_bit(queens, from);

        bitboard attacks = get_queen_attacks(1ULL << from, b);
        bitboard non_captures = attacks & ~(b.white_pieces | b.black_pieces);
        bitboard captures = attacks & b.white_pieces;

        while (non_captures) {
            int to = __builtin_ctzll(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), l);
        }

        while (captures) {
            int to = __builtin_ctzll(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), l);
        }
    }
}
