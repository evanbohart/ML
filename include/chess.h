#ifndef CHESS_H
#define CHESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define FILE_A 0x0101010101010101
#define FILE_B 0x0202020202020202
#define FILE_C 0x0404040404040404
#define FILE_D 0x0808080808080808
#define FILE_E 0x1010101010101010
#define FILE_F 0x2020202020202020
#define FILE_G 0x4040404040404040
#define FILE_H 0x8080808080808080
#define RANK_1 0x00000000000000FF
#define RANK_2 0x000000000000FF00
#define RANK_3 0x0000000000FF0000
#define RANK_4 0x00000000FF000000
#define RANK_5 0x000000FF00000000
#define RANK_6 0x0000FF0000000000
#define RANK_7 0x00FF000000000000
#define RANK_8 0xFF00000000000000


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

#define BITBOARD_ZERO ((bitboard)0)
#define BITBOARD_ONE ((bitboard)1)

void draw_bitboard(bitboard b);

#define set_bit(b, pos) ((b) |= (BITBOARD_ONE << (pos)))
#define set_bits(b, bits) ((b) |= (bits))
#define clear_bit(b, pos) ((b) &= ~(BITBOARD_ONE << (pos)))
#define clear_bits(b, bits) ((b) &= ~(bits))
#define check_bit(b, pos) ((b) & (BITBOARD_ONE << (pos)))
#define check_bits(b, bits) (((b) & (bits)) == (bits))

#define clear_from(b, pos) ((b) &= ((pos) >= 63 ? \
                           ~BITBOARD_ZERO : (((BITBOARD_ONE << ((pos) + 1)) - 1))))
#define clear_until(b, pos) ((b) &= ((pos) == 0 ? \
                            ~BITBOARD_ZERO : (~((BITBOARD_ONE << (pos)) - 1))))

#define calc_shift(b, x, y) (((x) + 8 * (y) > 0) ? \
                            (b) << ((x) + 8 * (y)) : \
                            (b) >> (-(x) - 8 * (y)))

typedef uint16_t move;

#define move_from(move) ((move) & 0x003F)
#define move_to(move) (((move) >> 6) & 0x003F)
#define move_flag(move) (((move) >> 12) & 0x000F)
#define create_move(from, to, flag) (((move)(from)) | ((move)((to) << 6)) | ((move)((flag) << 12)))

typedef enum flag { QUIET, DOUBLE_PUSH, CASTLE_SHORT, CASTLE_LONG, CAPTURE, EN_PASSANT,
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

enum { SHORT, LONG };

typedef struct board {
    bitboard pieces[2][6];
    bitboard pieces_color[2];
    bitboard pieces_all;

    bitboard attacks[2][6];
    bitboard attacks_all[2];

    bitboard pins_vertical[2];
    bitboard pins_horizontal[2];
    bitboard pins_diagonal1[2];
    bitboard pins_diagonal2[2];

    bitboard en_passant[2];

    bool castling_rights[2][2];

    piece piece_lookup[64];

    move_list legal_moves;
} board;

#define STARTING_WHITE_KING 0x0000000000000010
#define STARTING_WHITE_PAWNS 0x000000000000ff00
#define STARTING_WHITE_KNIGHTS 0x0000000000000042
#define STARTING_WHITE_BISHOPS 0x0000000000000024
#define STARTING_WHITE_ROOKS 0x0000000000000081
#define STARTING_WHITE_QUEENS 0x0000000000000008
#define STARTING_BLACK_KING 0x1000000000000000
#define STARTING_BLACK_PAWNS 0x00ff000000000000
#define STARTING_BLACK_KNIGHTS 0x4200000000000000
#define STARTING_BLACK_BISHOPS 0x2400000000000000
#define STARTING_BLACK_ROOKS 0x8100000000000000
#define STARTING_BLACK_QUEENS 0x0800000000000000

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

extern const bitboard bishop_rays[64][4];
extern const bitboard rook_rays[64][4];

void get_pins(board *b, color c);
void get_pins_vertical(board *b, color c);
void get_pins_horizontal(board *b, color c);
void get_pins_diagonal1(board *b, color c);
void get_pins_diagonal2(board *b, color c);

extern const bitboard bishop_masks[64];
extern const bitboard bishop_magic[64];
extern const int bishop_shifts[64];

extern const bitboard rook_masks[64];
extern const bitboard rook_magic[64];
extern const int rook_shifts[64];

int bishop_hash(bitboard blockers, int pos);
int rook_hash(bitboard blockers, int pos);

extern const bitboard knight_attack_table[64];
extern bitboard *bishop_attack_table[64];
extern bitboard *rook_attack_table[64];

void init_attack_tables(void);
void init_bishop_attack_table(void);
void init_rook_attack_table(void);

bitboard get_blockers(bitboard mask, int x);
void precompute_bishop_attacks(int pos);
void precompute_rook_attacks(int pos);

void destroy_attack_tables(void);
void destroy_bishop_attack_table(void);
void destroy_rook_attack_table(void);

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
