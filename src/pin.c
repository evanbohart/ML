#include "chess.h"

bitboard bishop_rays[64][4];
bitboard rook_rays[64][4];

void init_bishop_rays(void)
{
    for (int i = 0; i < 64; ++i) {
        bitboard blockers_nw = 0ULL;
        bitboard blockers_ne = 0ULL;
        bitboard blockers_se = 0ULL;
        bitboard blockers_sw = 0ULL;

        int file = i % 8;
        int rank = i / 8;

        if (file - 1 >= 0 && rank - 1 >= 0) {
            set_bit(blockers_nw, i - 9);
            set_bit(blockers_ne, i - 9);
            set_bit(blockers_se, i - 9);
        }

        if (file - 1 < 8 && rank - 1 >= 0) {
            set_bit(blockers_nw, i - 7);
            set_bit(blockers_ne, i - 7);
            set_bit(blockers_sw, i - 7);
        }

        if (file - 1 < 8 && rank - 1 < 8) {
            set_bit(blockers_nw, i + 9);
            set_bit(blockers_se, i + 9);
            set_bit(blockers_sw, i + 9);
        }

        if (file - 1 >= 0 && rank - 1 < 8) {
            set_bit(blockers_ne, i + 7);
            set_bit(blockers_se, i + 7);
            set_bit(blockers_sw, i + 7);
        }

        bishop_rays[i][0] = get_bishop_attacks_slow(blockers_nw, i) & ~blockers_nw;
        bishop_rays[i][1] = get_bishop_attacks_slow(blockers_ne, i) & ~blockers_ne;
        bishop_rays[i][2] = get_bishop_attacks_slow(blockers_se, i) & ~blockers_se;
        bishop_rays[i][3] = get_bishop_attacks_slow(blockers_sw, i) & ~blockers_sw;
    }
}

void init_rook_rays(void)
{
    for (int i = 0; i < 64; ++i) {
        bitboard blockers_up = 0ULL;
        bitboard blockers_down = 0ULL;
        bitboard blockers_left = 0ULL;
        bitboard blockers_right = 0ULL;

        int file = i % 8;
        int rank = i / 8;

        if (file - 1 >= 0) {
            set_bit(blockers_up, i - 1);
            set_bit(blockers_down, i - 1);
            set_bit(blockers_right, i - 1);
        }

        if (file - 1 < 8) {
            set_bit(blockers_up, i + 1);
            set_bit(blockers_down, i + 1);
            set_bit(blockers_left, i + 1);
        }

        if (rank - 1 >= 0) {
            set_bit(blockers_up, i - 8);
            set_bit(blockers_left, i - 8);
            set_bit(blockers_right, i - 8);
        }

        if (rank - 1 < 8) {
            set_bit(blockers_down, i + 8);
            set_bit(blockers_left, i + 8);
            set_bit(blockers_right, i + 8);
        }

        rook_rays[i][0] = get_rook_attacks_slow(blockers_up, i) & ~blockers_up;
        rook_rays[i][1] = get_rook_attacks_slow(blockers_down, i) & ~blockers_down;
        rook_rays[i][2] = get_rook_attacks_slow(blockers_left, i) & ~blockers_left;
        rook_rays[i][3] = get_rook_attacks_slow(blockers_right, i) & ~blockers_right;
    }
}

void get_white_pins(board *b)
{
    bitboard orthogonal_attackers = b->black_pieces[ROOK] | b->black_pieces[QUEEN];
    bitboard diagonal_attackers = b->black_pieces[BISHOP] | b->black_pieces[QUEEN];

    b->white_pins_vertical = get_pins_vertical(b->white_pieces[KING], orthogonal_attackers,
                                               b->white_pieces_all, b->pieces_all);
    b->white_pins_horizontal = get_pins_horizontal(b->white_pieces[KING], orthogonal_attackers,
                                                   b->white_pieces_all, b->pieces_all);
    b->white_pins_diagonal1 = get_pins_diagonal1(b->white_pieces[KING], diagonal_attackers,
                                                 b->white_pieces_all, b->pieces_all);
    b->white_pins_diagonal2 = get_pins_diagonal2(b->white_pieces[KING], diagonal_attackers,
                                                 b->white_pieces_all, b->pieces_all);
}

void get_black_pins(board *b)
{
    bitboard orthogonal_attackers = b->white_pieces[ROOK] | b->white_pieces[QUEEN];
    bitboard diagonal_attackers = b->white_pieces[BISHOP] | b->white_pieces[QUEEN];

    b->black_pins_vertical = get_pins_vertical(b->black_pieces[KING], orthogonal_attackers,
                                               b->black_pieces_all, b->pieces_all);
    b->black_pins_horizontal = get_pins_horizontal(b->black_pieces[KING], orthogonal_attackers,
                                                   b->black_pieces_all, b->pieces_all);
    b->black_pins_diagonal1 = get_pins_diagonal1(b->black_pieces[KING], diagonal_attackers,
                                                 b->black_pieces_all, b->pieces_all);
    b->black_pins_diagonal2 = get_pins_diagonal2(b->black_pieces[KING], diagonal_attackers,
                                                 b->black_pieces_all, b->pieces_all);
}

bitboard get_pins_vertical(bitboard king, bitboard attackers, bitboard friendly, bitboard pieces_all)
{
    bitboard pins = 0ULL;

    int king_pos = ctz(king);

    bitboard ray_up = rook_rays[king_pos][0];
    bitboard ray_down = rook_rays[king_pos][1];
    bitboard attackers_up = attackers & ray_up;
    bitboard attackers_down = attackers & ray_down;

    if (attackers_up) {
        int attacker_pos = ctz(attackers_up);
        bitboard pieces_between = pieces_all & ray_up & ((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    if (attackers_down) {
        int attacker_pos = 63 - clz(attackers_down);
        bitboard pieces_between = pieces_all & ray_down & ~((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    return pins;
}

bitboard get_pins_horizontal(bitboard king, bitboard attackers, bitboard friendly, bitboard pieces_all)
{
    bitboard pins = 0ULL;

    int king_pos = ctz(king);

    bitboard ray_left = rook_rays[king_pos][2];
    bitboard ray_right = rook_rays[king_pos][3];
    bitboard attackers_left = attackers & ray_left;
    bitboard attackers_right = attackers & ray_right;

    if (attackers_left) {
        int attacker_pos = 63 - clz(attackers_left);
        bitboard pieces_between = pieces_all & ray_left & ~((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    if (attackers_right) {
        int attacker_pos = ctz(attackers_right);
        bitboard pieces_between = pieces_all & ray_right & ((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    return pins;
}

bitboard get_pins_diagonal1(bitboard king, bitboard attackers, bitboard friendly, bitboard pieces_all)
{
    bitboard pins = 0ULL;

    int king_pos = ctz(king);

    bitboard ray_nw = bishop_rays[king_pos][0];
    bitboard ray_sw = bishop_rays[king_pos][3];
    bitboard attackers_nw = attackers & ray_nw;
    bitboard attackers_sw = attackers & ray_sw;

    if (attackers_nw) {
        int attacker_pos = ctz(attackers_nw);
        bitboard pieces_between = pieces_all & ray_nw & ((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    if (attackers_sw) {
        int attacker_pos = 63 - clz(attackers_sw);
        bitboard pieces_between = pieces_all & ray_sw & ~((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    return pins;
}

bitboard get_pins_diagonal2(bitboard king, bitboard attackers, bitboard friendly, bitboard pieces_all)
{
    bitboard pins = 0ULL;

    int king_pos = ctz(king);

    bitboard ray_ne = bishop_rays[king_pos][1];
    bitboard ray_se = bishop_rays[king_pos][2];
    bitboard attackers_ne = attackers & ray_ne;
    bitboard attackers_se = attackers & ray_se;

    if (attackers_ne) {
        int attacker_pos = ctz(attackers_ne);
        bitboard pieces_between = pieces_all & ray_ne & ((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    if (attackers_se) {
        int attacker_pos = 63 - clz(attackers_se);
        bitboard pieces_between = pieces_all & ray_se & ~((1ULL << attacker_pos) - 1);

        if (popcount(pieces_between) == 1 && check_bits(friendly, pieces_between)) {
            set_bits(pins, pieces_between);
        }
    }

    return pins;
}
