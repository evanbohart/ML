#ifndef CHESS_H
#define CHESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

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
#define clear_from(b, pos) ((b) &= ((1ULL << (pos)) - 1))
#define clear_until(b, pos) ((b) &= ~((1ULL << (pos)) - 1))
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
#define add_move(m, ml) do { \
                            assert((ml).count != MAX_MOVES); \
                            (ml).moves[(ml).count++] = m; \
                        } while (0)

void display_moves(move_list l);

typedef enum color { WHITE, BLACK } color;

typedef enum piece { KING, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, NONE } piece;

typedef struct board {
    bitboard pieces[2][6];
    bitboard attacks[2][6];
    bitboard pieces_all[2];
    bitboard attacks_all[2];
    bitboard pins_vertical[2];
    bitboard pins_horizontal[2];
    bitboard pins_diagonal1[2];
    bitboard pins_diagonal2[2];

    piece piece_lookup[64];

    move_list legal_moves;
} board;

#define total_occupancy(b) ((b).pieces_all[BLACK] | (b).pieces_all[WHITE])

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
void apply_move(board *b, color c, move m);
void update_board(board *b, color c);
bool check(board *b, color c);
bool checkmate(board *b, color c);
void draw_board(board *b);

void get_legal_moves(board *b, color c);
void get_king_moves(board *b, color c);
void get_pawn_moves(board *b, color c);
void get_knight_moves(board *b, color c);
void get_bishop_moves(board *b, color c);
void get_queen_moves(board *b, color c);
void get_rook_moves(board *b, color c);

extern bitboard bishop_rays[64][4];
extern bitboard rook_rays[64][4];

void init_bishop_rays(void);
void init_rook_rays(void);

void get_pins(board *b, color c);
void get_pins_vertical(board *b, color c);
void get_pins_horizontal(board *b, color c);
void get_pins_diagonal1(board *b, color c);
void get_pins_diagonal2(board *b, color c);

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

void get_attacks(board *b, color c);
void get_king_attacks(board *b, color c);
void get_pawn_attacks(board *b, color c);
void get_knight_attacks(board *b, color c);
void get_bishop_attacks(board *b, color c);
void get_rook_attacks(board *b, color c);
void get_queen_attacks(board *b, color c);

#ifdef __cplusplus
}
#endif

#endif
