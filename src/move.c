#include <stdio.h>
#include "chess.h"

void display_moves(move_list ml)
{
    for (int i = 0; i < ml.count; ++i) {
        int from = move_from(ml.moves[i]);
        int to = move_to(ml.moves[i]);
        int flag = move_flag(ml.moves[i]);

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

void get_legal_moves(board *b, color c)
{
    clear_moves(b->legal_moves);

    get_king_moves(b, c);
    get_pawn_moves(b, c);
    get_knight_moves(b, c);
    get_bishop_moves(b, c);
    get_rook_moves(b, c);
    get_queen_moves(b, c);
}

void get_king_moves(board *b, color c)
{
    bitboard attacks = b->attacks[c][KING];
    bitboard non_captures = attacks & ~b->attacks_all[!c] & ~total_occupancy(*b);
    bitboard captures = attacks & ~b->attacks_all[!c] & b->pieces_all[!c];

    int from = ctz(b->pieces[c][KING]);

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

void get_pawn_moves(board *b, color c)
{
    int dir = c ? 1 : -1;
    bitboard push_pins = b->pins_horizontal[c] | b->pins_diagonal1[c] | b->pins_diagonal2[c];
    bitboard push_pawns = b->pieces[c][PAWN] & ~push_pins;
    bitboard single_pushes = calc_shift(push_pawns, 0, dir) & ~total_occupancy(*b);
    bitboard double_pushes = calc_shift(single_pushes, 0, dir) &
                             ~total_occupancy(*b) & (c ? MASK_4 : MASK_5);
    bitboard promotions = single_pushes & (c ? MASK_8 : MASK_1);
    bitboard quiet_pushes = single_pushes & ~(c ? MASK_8 : MASK_1);

    while (quiet_pushes) {
        int to = ctz(quiet_pushes);
        clear_bit(quiet_pushes, to);
        int from = to - 8 * dir;

        add_move(create_move(from, to, QUIET), b->legal_moves);
    }

    while (double_pushes) {
        int to = ctz(double_pushes);
        clear_bit(double_pushes, to);
        int from = to - 16 * dir;

        add_move(create_move(from, to, DOUBLE_PUSH), b->legal_moves);
    }

    while (promotions) {
        int to = ctz(promotions);
        clear_bit(promotions, to);
        int from = to - 8 * dir;

        add_move(create_move(from, to, PROMO_KNIGHT), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN), b->legal_moves);
    }

    bitboard capture_pins = b->pins_vertical[c] | b->pins_horizontal[c];
    bitboard capture_pawns = b->pieces[c][PAWN] & ~capture_pins;
    bitboard potential_captures = b->attacks[2][PAWN] & b->pieces_all[c];
    bitboard captures_left = potential_captures & calc_shift(capture_pawns & ~b->pins_diagonal2[c], -1, dir);
    bitboard captures_right = potential_captures & calc_shift(capture_pawns & ~b->pins_diagonal1[c], 1, dir);
    bitboard promotions_left = captures_left & (c ? MASK_8 : MASK_1);
    bitboard promotions_right = captures_right & (c ? MASK_8 : MASK_1);
    bitboard non_promotions_left = captures_left & ~(c ? MASK_8 : MASK_1);
    bitboard non_promotions_right = captures_right & ~(c ? MASK_8 : MASK_1);

    while (non_promotions_left) {
        int to = ctz(non_promotions_left);
        clear_bit(non_promotions_left, to);
        int from = to - 7 * dir;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (non_promotions_right) {
        int to = ctz(non_promotions_right);
        clear_bit(non_promotions_right, to);
        int from = to - 9 * dir;

        add_move(create_move(from, to, CAPTURE), b->legal_moves);
    }

    while (promotions_left) {
        int to = ctz(promotions_left);
        clear_bit(promotions_left, to);
        int from = to - 7 * dir;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }

    while (promotions_right) {
        int to = ctz(promotions_right);
        clear_bit(promotions_right, to);
        int from = to - 9 * dir;

        add_move(create_move(from, to, PROMO_KNIGHT_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_BISHOP_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_ROOK_CAPTURE), b->legal_moves);
        add_move(create_move(from, to, PROMO_QUEEN_CAPTURE), b->legal_moves);
    }
}

void get_knight_moves(board *b, color c)
{/**
    TODO FIGURE THIS OUT
    bitboard pins = b->pins_vertical[c] | b->pins_horizontal[c] |
                    b->pins_diagonal1[c] | b->pins_diagonal2[c];
    bitboard knights = b->pieces[c][KNIGHT] & ~pins;

    while (knights) {
        int from = ctz(knights);
        clear_bit(knights, from);

        bitboard attacks =;
        bitboard non_captures = attacks & ~total_occupancy(*b);
        bitboard captures = attacks & b->pieces_all[!c];

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
    }*/
}

void get_bishop_moves(board *b, color c)
{
    bitboard pins = b->pins_vertical[c] | b->pins_vertical[c];
    bitboard bishops = b->pieces[c][BISHOP] & ~pins;

    while (bishops) {
        int from = ctz(bishops);
        clear_bit(bishops, from);

        bitboard ray_diagonal1 = bishop_rays[from][0] | bishop_rays[from][3];
        bitboard ray_diagonal2 = bishop_rays[from][1] | bishop_rays[from][2];

        int index = bishop_hash(total_occupancy(*b), from);
        bitboard attacks = rook_attack_table[from][index];

        bitboard attacks_diagonal1 = check_bit(b->pins_diagonal2[c], from) ? 0ULL : attacks & ray_diagonal1;
        bitboard attacks_diagonal2 = check_bit(b->pins_diagonal1[c], from) ? 0ULL : attacks & ray_diagonal2;
        bitboard non_captures_diagonal1 = attacks_diagonal1 & ~total_occupancy(*b);
        bitboard non_captures_diagonal2 = attacks_diagonal2 & ~total_occupancy(*b);
        bitboard captures_diagonal1 = attacks_diagonal1 & b->pieces_all[!c];
        bitboard captures_diagonal2 = attacks_diagonal2 & b->pieces_all[!c];
        bitboard non_captures = non_captures_diagonal1 | non_captures_diagonal2;
        bitboard captures = captures_diagonal1 | captures_diagonal2;

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

void get_rook_moves(board *b, color c)
{
    bitboard pins = b->pins_diagonal1[c] | b->pins_diagonal2[c];
    bitboard rooks = b->pieces[c][ROOK] & ~pins;

    while (rooks) {
        int from = ctz(rooks);
        clear_bit(rooks, from);

        bitboard ray_vertical = rook_rays[from][0] | rook_rays[from][1];
        bitboard ray_horizontal = rook_rays[from][2] | rook_rays[from][3];

        int index = rook_hash(total_occupancy(*b), from);
        bitboard attacks = rook_attack_table[from][index];

        bitboard attacks_vertical = check_bit(b->pins_horizontal[c], from) ? 0ULL : attacks & ray_vertical;
        bitboard attacks_horizontal = check_bit(b->pins_vertical[c], from) ? 0ULL : attacks & ray_horizontal;
        bitboard non_captures_vertical = attacks_vertical & ~total_occupancy(*b);
        bitboard non_captures_horizontal = attacks_horizontal & ~total_occupancy(*b);
        bitboard captures_vertical = attacks_vertical & b->pieces_all[!c];
        bitboard captures_horizontal = attacks_horizontal & b->pieces_all[!c];
        bitboard non_captures = non_captures_vertical | non_captures_horizontal;
        bitboard captures = captures_vertical | captures_horizontal;

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

void get_queen_moves(board *b, color c)
{
    bitboard queens = b->pieces[c][QUEEN];

    while (queens) {
        int from = ctz(queens);
        clear_bit(queens, from);

        bitboard ray_vertical = rook_rays[from][0] | rook_rays[from][1];
        bitboard ray_horizontal = rook_rays[from][2] | rook_rays[from][3];
        bitboard ray_diagonal1 = bishop_rays[from][0] | bishop_rays[from][3];
        bitboard ray_diagonal2 = bishop_rays[from][1] | bishop_rays[from][2];

        int bishop_table_index = bishop_hash(total_occupancy(*b), from);
        int rook_table_index = rook_hash(total_occupancy(*b), from);
        bitboard attacks = bishop_attack_table[from][bishop_table_index] |
                           rook_attack_table[from][rook_table_index];

        bitboard attacks_vertical = check_bit(b->pins_horizontal[c] | b->pins_diagonal1[c] |
                                              b->pins_diagonal2[c], from) ? 0ULL : attacks & ray_vertical;
        bitboard attacks_horizontal = check_bit(b->pins_vertical[c] | b->pins_diagonal1[c] |
                                                b->pins_diagonal2[c], from) ? 0ULL : attacks & ray_horizontal;
        bitboard attacks_diagonal1 = check_bit(b->pins_vertical[c] | b->pins_horizontal[c] |
                                               b->pins_diagonal2[c], from) ? 0ULL : attacks & ray_diagonal1;
        bitboard attacks_diagonal2 = check_bit(b->pins_vertical[c] | b->pins_horizontal[c] |
                                               b->pins_diagonal1[c], from) ? 0ULL : attacks & ray_diagonal2;
        bitboard non_captures_vertical = attacks_vertical & ~total_occupancy(*b);
        bitboard non_captures_horizontal = attacks_horizontal & ~total_occupancy(*b);
        bitboard non_captures_diagonal1 = attacks_diagonal1 & ~total_occupancy(*b);
        bitboard non_captures_diagonal2 = attacks_diagonal2 & ~total_occupancy(*b);
        bitboard captures_vertical = attacks_vertical & b->pieces_all[!c];
        bitboard captures_horizontal = attacks_horizontal & b->pieces_all[!c];
        bitboard captures_diagonal1 = attacks_diagonal1 & b->pieces_all[!c];
        bitboard captures_diagonal2 = attacks_diagonal2 & b->pieces_all[!c];
        bitboard non_captures = non_captures_vertical | non_captures_horizontal |
                                non_captures_diagonal1 | non_captures_diagonal2;
        bitboard captures = captures_vertical | captures_horizontal |
                            captures_diagonal1 | captures_diagonal2;

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
