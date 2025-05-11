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


bitboard get_bishop_attacks_slow(bitboard blockers, int pos)
{
    bitboard attacks = 0ULL;

    int file = pos % 8;
    int rank = pos / 8;

    for (int i = file + 1, j = rank + 1; i < 8 && j < 8; ++i, ++j) {
        set_bit(&attacks, j * 8 + i);
        if (check_bit(blockers, j * 8 + i)) break;
    }

    for (int i = file + 1, j = rank - 1; i < 8 && j >= 0; ++i, --j) {
        set_bit(&attacks, j * 8 + i);
        if (check_bit(blockers, j * 8 + i)) break;
    }

    for (int i = file - 1, j = rank - 1; i >= 0 && j >= 0; --i, --j) {
        set_bit(&attacks, j * 8 + i);
        if (check_bit(blockers, j * 8 + i)) break;
    }

    for (int i = file - 1, j = rank + 1; i >= 0 && j < 8; --i, ++j) {
        set_bit(&attacks, j * 8 + i);
        if (check_bit(blockers, j * 8 + i)) break;
    }

    return attacks;
}

void init_bishop_attack_table(void)
{
    for (int i = 0; i < 64; ++i) {
        int n = 1 << (64 - bishop_shifts[i]);
        bishop_attack_table[i] = malloc(n * sizeof(bitboard));
        bitboard blockers[n];
        bitboard temp = 0ULL;

        for (int j = 0; j < n; ++j) {
            bool fail = true;
            while (fail) {
                fail = false;
                blockers[j] = apply_masks(rand_bitboard(&temp, 32), 1, bishop_masks[i]);
                temp = 0ULL;
                for (int k = 0; k < j; ++k) {
                    if (blockers[j] == blockers[k]) {
                        fail = true;
                        break;
                    }
                }
            }

            int index = bishop_hash(blockers[j], bishop_shifts[i]);
            bishop_attack_table[i][index] = get_bishop_attacks_slow(blockers[j], i);
        }
    }
}

bitboard *bishop_attack_table[64];

const bitboard bishop_masks[64] = {
    0x0902000400004902ULL, 0x2000044000000010ULL,
    0x0800031508000400ULL, 0x200000000a002300ULL,
    0x4600024018000102ULL, 0x1000004040480100ULL,
    0x0000010a20103000ULL, 0x010100002a440008ULL,
    0x084c480040000290ULL, 0x000e001102441a24ULL,
    0x2100100000000000ULL, 0x4000188020200004ULL,
    0x0020012004020500ULL, 0x4080608100020620ULL,
    0x0011224000000000ULL, 0x6002020300020200ULL,
    0x4200080040000400ULL, 0x0000000110090000ULL,
    0x0010403848000000ULL, 0x0000110040202410ULL,
    0x020020c000804d04ULL, 0x0034000024062000ULL,
    0x4404000004010020ULL, 0x0000308000000010ULL,
    0x0002004000000108ULL, 0x2421010000100144ULL,
    0x0046010802205080ULL, 0x0010000440600000ULL,
    0x0048004053030280ULL, 0x2302200020040008ULL,
    0x1000102000842284ULL, 0x0002000020081014ULL,
    0x10d8000040160000ULL, 0x0608002205105001ULL,
    0x1000000040000108ULL, 0x4020480120800000ULL,
    0x022028005c040840ULL, 0x0000000070204041ULL,
    0x000000060b884064ULL, 0x00010000011008a6ULL,
    0x2080044500080024ULL, 0x4000000250001040ULL,
    0x0002004802830000ULL, 0x05000000000a00a0ULL,
    0x0180000002000000ULL, 0x0474302641000600ULL,
    0x2000020000004840ULL, 0x2012400400000000ULL,
    0x0402400120410000ULL, 0x0700001800002804ULL,
    0x0000240000000083ULL, 0x0840212100400000ULL,
    0x408808020012008aULL, 0x088821dc00000800ULL,
    0x0510000000200c01ULL, 0x0c0000c002200420ULL,
    0x0825008020810450ULL, 0x2400209400021100ULL,
    0x00000200000c0040ULL, 0x20011403200005a2ULL,
    0x0000040200010200ULL, 0x0004000428010001ULL,
    0x2005080440001400ULL, 0x0080401a00160d00ULL
};

const bitboard bishop_magic[64] = {
    0x1010002805000084ULL, 0x0000100410806020ULL,
    0x0001480214080102ULL, 0x0240080805020103ULL,
    0x0000190000840900ULL, 0x0810018200024020ULL,
    0x2090404220000000ULL, 0x0a40009248800802ULL,
    0x00814040400000c1ULL, 0x000005c442800100ULL,
    0x0010000100000010ULL, 0x0014020806001050ULL,
    0x0410000000000129ULL, 0x0000161010422000ULL,
    0x02001080049020acULL, 0x0148400006004800ULL,
    0x4098080002400000ULL, 0x4400108040000c02ULL,
    0x0008000000030024ULL, 0x22900000102a0000ULL,
    0x0000440032000000ULL, 0x0000090102000004ULL,
    0x0240400120100000ULL, 0x404040a00b900880ULL,
    0x0003200000012000ULL, 0x1020140048a80000ULL,
    0x200200d400400002ULL, 0x0000481120044080ULL,
    0x2004459001081040ULL, 0x2058220020000202ULL,
    0x2000000800040200ULL, 0x0c00020200000001ULL,
    0x110c000001080041ULL, 0x0260000701800000ULL,
    0x0001204108001051ULL, 0x021c000000080801ULL,
    0x0400090000050000ULL, 0x0900010140344800ULL,
    0x1000212806000200ULL, 0x0421000204280408ULL,
    0x088200a000002300ULL, 0x00a0000130101000ULL,
    0x0000001200200028ULL, 0x1195482000000008ULL,
    0x070a020206060804ULL, 0x00a4000000000000ULL,
    0x0340000000200019ULL, 0x2041048007004410ULL,
    0x0201000061044a00ULL, 0x0034000120001100ULL,
    0x0100000030080090ULL, 0x0008040000204001ULL,
    0x0810000040000801ULL, 0x2001000205106040ULL,
    0x4080210004002420ULL, 0x0901441504121100ULL,
    0x0000000008010008ULL, 0x0100006000222400ULL,
    0x0085000860400610ULL, 0x0000400800000000ULL,
    0x2480000008020120ULL, 0x400000000005100cULL,
    0x300140080c210002ULL, 0x20220540102d0c20ULL
};

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
    return apply_masks(blockers, 1, bishop_masks[pos]) * bishop_magic[pos] >> bishop_shifts[pos];
}

int rook_hash(bitboard blockers, int pos)
{
    return 0;
}
