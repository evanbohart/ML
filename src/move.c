#include <stdio.h>
#include <assert.h>
#include "chess.h"

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

void get_white_moves(board *b)
{
    clear_moves(b->legal_moves);

    get_white_king_moves(b);
    get_white_pawn_moves(b);
    get_white_knight_moves(b);
    get_white_bishop_moves(b);
    get_white_rook_moves(b);
    get_white_queen_moves(b);
}

void get_black_moves(board *b)
{
    clear_moves(b->legal_moves);

    get_black_king_moves(b);
    get_black_pawn_moves(b);
    get_black_knight_moves(b);
    get_black_bishop_moves(b);
    get_black_rook_moves(b);
    get_black_queen_moves(b);
}

void get_white_king_moves(board *b)
{
    bitboard attacks = get_king_attacks(b->white_pieces[KING]);
    bitboard non_captures = attacks & ~b->black_attacks_all & ~b->pieces_all;
    bitboard captures = attacks & ~b->black_attacks_all & b->black_pieces_all;

    int from = ctz(b->white_pieces[KING]);

    while (non_captures) {
        int to = ctz(non_captures);
        clear_bit(non_captures, to);

        add_move(create_move(from, to, QUIET), b->legal_moves);
    }

    while (captures) {
        int to = ctz(captures);
        clear_bit(captures, to);

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }
}

void get_black_king_moves(board *b)
{
    bitboard attacks = b->black_attacks[KING];
    bitboard non_captures = attacks & ~b->white_attacks_all & ~b->pieces_all;
    bitboard captures = attacks & ~b->white_attacks_all & b->white_pieces_all;

    int from = ctz(b->black_pieces[KING]);

    while (non_captures) {
        int to = ctz(non_captures);
        clear_bit(non_captures, to);

        add_move(create_move(from, to, QUIET), b->legal_moves);
    }

    while (captures) {
        int to = ctz(captures);
        clear_bit(captures, to);

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }
}

void get_white_pawn_moves(board *b)
{
    bitboard push_pins = b->white_pins_horizontal | b->white_pins_diagonal1 | b->white_pins_diagonal2;
    bitboard push_pawns = b->white_pieces[PAWN] & ~push_pins;
    bitboard single_pushes = calc_shift(push_pawns, 0, 1) & ~b->pieces_all;
    bitboard double_pushes = calc_shift(single_pushes, 0, 1) & ~b->pieces_all & MASK_4;
    bitboard promotions = single_pushes & MASK_8;
    bitboard quiet_pushes = single_pushes & ~MASK_8;

    while (quiet_pushes) {
        int to = ctz(quiet_pushes);
        clear_bit(quiet_pushes, to);
        int from = to - 8;

        add_move(create_move(from, to, QUIET), b->legal_moves);
    }

    while (double_pushes) {
        int to = ctz(double_pushes);
        clear_bit(double_pushes, to);
        int from = to - 16;

        add_move(create_move(from, to, DOUBLE_PUSH), b->legal_moves);
    }

    while (promotions) {
        int to = ctz(promotions);
        clear_bit(promotions, to);
        int from = to - 8;

        add_move(create_move(from, to, PROMO_KNIGHT), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN), b->legal_moves);
    }

    bitboard capture_pins = b->white_pins_vertical | b->white_pins_horizontal;
    bitboard capture_pawns = b->white_pieces[PAWN] & ~capture_pins;
    bitboard potential_captures = b->white_attacks[PAWN] & b->black_pieces_all;
    bitboard captures_left = potential_captures & calc_shift(capture_pawns & ~b->white_pins_diagonal2, -1, 1);
    bitboard captures_right = potential_captures & calc_shift(capture_pawns & ~b->white_pins_diagonal1, 1, 1);
    bitboard promotions_left = captures_left & ~MASK_8;
    bitboard promotions_right = captures_right & ~MASK_8;

    captures_left &= MASK_8;
    captures_right &= MASK_8;

    while (captures_left) {
        int to = ctz(captures_left);
        clear_bit(captures_left, to);
        int from = to - 7;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (captures_right) {
        int to = ctz(captures_right);
        clear_bit(captures_right, to);
        int from = to - 9;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (promotions_left) {
        int to = ctz(promotions_left);
        clear_bit(promotions_left, to);
        int from = to - 7;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }

    while (promotions_right) {
        int to = ctz(promotions_right);
        clear_bit(promotions_right, to);
        int from = to - 9;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }
}

void get_black_pawn_moves(board *b)
{
    bitboard push_pins = b->black_pins_horizontal | b->black_pins_diagonal1 | b->black_pins_diagonal2;
    bitboard push_pawns = b->black_pieces[PAWN] & ~push_pins;
    bitboard single_pushes = calc_shift(push_pawns, 0, -1) & ~b->pieces_all;
    bitboard double_pushes = calc_shift(single_pushes, 0, -1) & ~b->pieces_all & MASK_5;
    bitboard promotions = single_pushes & MASK_1;
    bitboard quiet_pushes = single_pushes & ~MASK_1;

    while (quiet_pushes) {
        int to = ctz(quiet_pushes);
        clear_bit(quiet_pushes, to);
        int from = to + 8;

        add_move(create_move(from, to, QUIET), b->legal_moves);
    }

    while (double_pushes) {
        int to = ctz(double_pushes);
        clear_bit(double_pushes, to);
        int from = to + 16;

        add_move(create_move(from, to, DOUBLE_PUSH), b->legal_moves);
    }

    while (promotions) {
        int to = ctz(promotions);
        clear_bit(promotions, to);
        int from = to + 8;

        add_move(create_move(from, to, PROMO_KNIGHT), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN), b->legal_moves);
    }

    bitboard capture_pins = b->black_pins_vertical | b->black_pins_horizontal;
    bitboard capture_pawns = b->black_pieces[PAWN] & ~capture_pins;
    bitboard potential_captures = b->black_attacks[PAWN] & b->white_pieces_all;
    bitboard captures_left = potential_captures & calc_shift(capture_pawns & ~b->black_pins_diagonal1, -1, -1);
    bitboard captures_right = potential_captures & calc_shift(capture_pawns & ~b->black_pins_diagonal2, 1, -1);
    bitboard promotions_left = captures_left & MASK_1;
    bitboard promotions_right = captures_right & MASK_1;

    captures_left &= ~MASK_1;
    captures_right  &= ~MASK_1;


    while (captures_left) {
        int to = ctz(captures_left);
        clear_bit(captures_left, to);
        int from = to + 9;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (captures_right) {
        int to = ctz(captures_right);
        clear_bit(captures_right, to);
        int from = to + 7;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (promotions_left) {
        int to = ctz(promotions_left);
        clear_bit(promotions_left, to);
        int from = to + 9;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }

    while (promotions_right) {
        int to = ctz(promotions_right);
        clear_bit(promotions_right, to);
        int from = to + 7;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }
}

void get_white_knight_moves(board *b)
{
    bitboard pins = b->white_pins_vertical | b->white_pins_horizontal |
                    b->white_pins_diagonal1 | b->white_pins_diagonal2;
    bitboard knights = b->white_pieces[KNIGHT] & ~pins;

    while (knights) {
        int from = ctz(knights);
        clear_bit(knights, from);

        bitboard attacks = get_knight_attacks(1ULL << from);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->black_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_black_knight_moves(board *b)
{
    bitboard pins = b->black_pins_vertical | b->black_pins_horizontal |
                    b->black_pins_diagonal1 | b->black_pins_diagonal2;
    bitboard knights = b->black_pieces[KNIGHT] & ~pins;

    while (knights) {
        int from = ctz(knights);
        clear_bit(knights, from);

        bitboard attacks = get_knight_attacks(1ULL << from);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->white_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_white_bishop_moves(board *b)
{
    bitboard pins = b->white_pins_vertical | b->white_pins_horizontal;
    bitboard bishops = b->white_pieces[BISHOP] & ~pins;

    while (bishops) {
        int from = ctz(bishops);
        clear_bit(bishops, from);

        bitboard attacks = get_bishop_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->black_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_black_bishop_moves(board *b)
{
    bitboard bishops = b->black_pieces[BISHOP];

    while (bishops) {
        int from = ctz(bishops);
        clear_bit(bishops, from);

        bitboard attacks = get_bishop_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->white_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_white_rook_moves(board *b)
{
    bitboard rooks = b->white_pieces[ROOK];

    while (rooks) {
        int from = ctz(rooks);
        clear_bit(rooks, from);

        bitboard attacks = get_rook_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->black_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_black_rook_moves(board *b)
{
    bitboard rooks = b->black_pieces[ROOK];

    while (rooks) {
        int from = ctz(rooks);
        clear_bit(rooks, from);

        bitboard attacks = get_rook_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->white_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_white_queen_moves(board *b)
{
    bitboard queens = b->white_pieces[QUEEN];

    while (queens) {
        int from = ctz(queens);
        clear_bit(queens, from);

        bitboard attacks = get_queen_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->black_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}

void get_black_queen_moves(board *b)
{
    bitboard queens = b->black_pieces[QUEEN];

    while (queens) {
        int from = ctz(queens);
        clear_bit(queens, from);

        bitboard attacks = get_queen_attacks(1ULL << from, b->pieces_all);
        bitboard non_captures = attacks & ~b->pieces_all;
        bitboard captures = attacks & b->white_pieces_all;

        while (non_captures) {
            int to = ctz(non_captures);
            clear_bit(non_captures, to);

            add_move(create_move(from, to, QUIET), b->legal_moves);
        }

        while (captures) {
            int to = ctz(captures);
            clear_bit(captures, to);

            add_move(create_move(from, to, CAPTURE), b->legal_moves);
        }
    }
}
