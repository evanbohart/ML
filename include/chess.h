#ifndef CHESS_H
#define CHESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

typedef enum col { A, B, C, D, E, F, G, H } col;

#define MASK_A 0x0101010101010101ULL
#define MASK_B 0x0202020202020202ULL
#define MASK_C 0x0404040404040404ULL
#define MASK_D 0x0808080808080808ULL
#define MASK_E 0x1010101010101010ULL
#define MASK_F 0x2020202020202020ULL
#define MASK_G 0x4040404040404040ULL
#define MASK_H 0x8080808080808080ULL
#define MASK_1 0x00000000000000FFULL
#define MASK_2 0x000000000000FF00ULL
#define MASK_3 0x0000000000FF0000ULL
#define MASK_4 0x00000000FF000000ULL
#define MASK_5 0x000000FF00000000ULL
#define MASK_6 0x0000FF0000000000ULL
#define MASK_7 0x00FF000000000000ULL
#define MASK_8 0xFF00000000000000ULL


#ifdef _WIN32
#define popcount __builtin_popcountll
#define ctz __builtin_ctzll
#define clz __builtin_clzll
#else
#define popcount __builtin_popcountl
#define ctz __builtin_ctzl
#define clz __builtin_clzl
#endif

typedef uint64_t bitboard;

void draw_bitboard(bitboard b);

#define set_bit(b, pos) ((b) |= (1ULL << (pos)))
#define set_bits(b, bits) ((b) |= (bits))
#define clear_bit(b, pos) ((b) &= ~(1ULL << (pos)))
#define clear_bits(b, bits) ((b) &= ~(bits))
#define check_bit(b, pos) ((b) & (1ULL << (pos)))
#define check_bits(b, bits) (((b) & (bits)) == (bits))

#define calc_shift(bitboard, x, y) (((x) + 8 * (y) > 0) ? \
                                   (bitboard) << ((x) + 8 * (y)) : \
                                   (bitboard) >> (-(x) - 8 * (y)))

typedef uint16_t move;

#define move_from(move) ((move) & 0x003F)
#define move_to(move) (((move) >> 6) & 0x003F)
#define move_flag(move) (((move) >> 12) & 0x000F)
#define create_move(from, to, flag) (((move)(from)) | ((move)((to) << 6)) | ((move)((flag) << 12)))

typedef enum flag1 { QUIET, DOUBLE_PUSH, SHORT_CASTLE, LONG_CASTLE, CAPTURE, ENPASSANT,
                    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN, PROMO_KNIGHT_CAPTURE,
                    PROMO_BISHOP_CAPTURE, PROMO_ROOK_CAPTURE, PROMO_QUEEN_CAPTURE } flag;

#define MAX_MOVES 256

typedef struct move_list {
    move moves[MAX_MOVES];
    int count;
} move_list;

#define clear_moves(ml) ((ml).count = 0)
#define add_move(m, ml) ((ml).moves[(ml).count++] = m)

void display_moves(move_list l);

typedef enum piece { KING, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, NONE } piece;

typedef struct board {
    bitboard white_pieces[6], black_pieces[6];
    bitboard white_pieces_all, black_pieces_all;
    bitboard pieces_all;

    bitboard white_attacks[6], black_attacks[6];
    bitboard white_attacks_all, black_attacks_all;

    bitboard white_pins_vertical, black_pins_vertical;
    bitboard white_pins_horizontal, black_pins_horizontal;
    bitboard white_pins_diagonal1, black_pins_diagonal1;
    bitboard white_pins_diagonal2, black_pins_diagonal2;

    piece piece_lookup[64];

    move_list legal_moves;
} board;

#define STARTING_WHITE_KING 0x0000000000000010ULL
#define STARTING_WHITE_PAWNS 0x000000000000ff00ULL
#define STARTING_WHITE_KNIGHTS 0x0000000000000042ULL
#define STARTING_WHITE_BISHOPS 0x0000000000000024ULL
#define STARTING_WHITE_ROOKS 0x0000000000000081ULL
#define STARTING_WHITE_QUEENS 0x0000000000000008ULL
#define STARTING_BLACK_KING 0x1000000000000000ULL
#define STARTING_BLACK_PAWNS 0x00ff000000000000ULL
#define STARTING_BLACK_KNIGHTS 0x4200000000000000ULL
#define STARTING_BLACK_BISHOPS 0x2400000000000000ULL
#define STARTING_BLACK_ROOKS 0x8100000000000000ULL
#define STARTING_BLACK_QUEENS 0x0800000000000000ULL

board init_board(void);
void init_piece_lookup(piece *piece_lookup);
void apply_move_white(board *b, move m);
void apply_move_black(board *b, move m);
void update_board_white(board *b);
void update_board_black(board *b);
bool white_check(board *b);
bool black_check(board *b);
bool white_checkmate(board *b);
bool black_checkmate(board *b);
void draw_board(board *b);

void get_white_moves(board *b);
void get_white_king_moves(board *b);
void get_white_pawn_moves(board *b);
void get_white_knight_moves(board *b);
void get_white_bishop_moves(board *b);
void get_white_queen_moves(board *b);
void get_white_rook_moves(board *b);

void get_black_moves(board *b);
void get_black_king_moves(board *b);
void get_black_pawn_moves(board *b);
void get_black_knight_moves(board *b);
void get_black_bishop_moves(board *b);
void get_black_queen_moves(board *b);
void get_black_rook_moves(board *b);

extern bitboard bishop_rays[64][4];
extern bitboard rook_rays[64][4];

void init_bishop_rays(void);
void init_rook_rays(void);

void get_white_pins(board *b);
void get_black_pins(board *b);

bitboard get_pins_vertical(bitboard king, bitboard attackers, bitboard friendly, bitboard all_pieces);
bitboard get_pins_horizontal(bitboard king, bitboard attackers, bitboard friendly, bitboard all_pieces);
bitboard get_pins_diagonal1(bitboard king, bitboard attackers, bitboard friendly, bitboard all_pieces);
bitboard get_pins_diagonal2(bitboard king, bitboard attackers, bitboard friendly, bitboard all_pieces);

void get_white_attacks(board *b);
void get_black_attacks(board *b);

bitboard get_king_attacks(bitboard king);
bitboard get_white_pawn_attacks(bitboard pawns);
bitboard get_black_pawn_attacks(bitboard pawns);
bitboard get_knight_attacks(bitboard knights);
bitboard get_bishop_attacks(bitboard bishops, bitboard all_pieces);
bitboard get_rook_attacks(bitboard rooks, bitboard all_pieces);
bitboard get_queen_attacks(bitboard queens, bitboard all_pieces);

extern bitboard *bishop_attack_table[64];

extern const bitboard bishop_masks[64];
extern const bitboard bishop_magic[64];
extern const int bishop_shifts[64];

extern bitboard *rook_attack_table[64];

extern const bitboard rook_masks[64];
extern const bitboard rook_magic[64];
extern const int rook_shifts[64];

bitboard get_bishop_attacks_slow(bitboard blockers, int pos);
bitboard get_rook_attacks_slow(bitboard blockers, int pos);

void init_bishop_attack_table(void);
void init_rook_attack_table(void);
void destroy_bishop_attack_table(void);
void destroy_rook_attack_table(void);

int bishop_hash(bitboard blockers, int pos);
int rook_hash(bitboard blockers, int pos);

bitboard get_blockers(bitboard mask, int x);

#ifdef __cplusplus
}
#endif

#endif
