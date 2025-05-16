#ifndef CHESS_H
#define CHESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

enum col { A, B, C, D, E, F, G, H };

#define MASK_A ~0x0101010101010101ULL
#define MASK_B ~0x0202020202020202ULL
#define MASK_C ~0x0404040404040404ULL
#define MASK_D ~0x0808080808080808ULL
#define MASK_E ~0x1010101010101010ULL
#define MASK_F ~0x2020202020202020ULL
#define MASK_G ~0x4040404040404040ULL
#define MASK_H ~0x8080808080808080ULL
#define MASK_1 ~0x00000000000000FFULL
#define MASK_2 ~0x000000000000FF00ULL
#define MASK_3 ~0x0000000000FF0000ULL
#define MASK_4 ~0x00000000FF000000ULL
#define MASK_5 ~0x000000FF00000000ULL
#define MASK_6 ~0x0000FF0000000000ULL
#define MASK_7 ~0x00FF000000000000ULL
#define MASK_8 ~0xFF00000000000000ULL

#define calc_shift(bitboard, x, y) (((x) + 8 * (y) > 0) ? \
                                   (bitboard) << ((x) + 8 * (y)) : \
                                   (bitboard) >> (-(x) - 8 * (y)))

typedef uint64_t bitboard;
typedef uint16_t move; // bits 0-5 is previous position, bits 6-11 is new position,
                       // bits 12-13 are flags such as castling, promotion, en passant

#define bitboard_at(col, row) (1ULL << (((row) - 1) * 8 + (col)))

bitboard rand_bitboard(bitboard *occupied, int c);
bitboard rand_pawns(bitboard *occupied);
bitboard rand_knights(bitboard *occupied);
bitboard rand_bishops(bitboard *occupied);
bitboard rand_rooks(bitboard *occupied);
bitboard rand_queens(bitboard *occupied);
void draw_bitboard(bitboard b);

static inline void set_bit(bitboard *b, int pos) { *b |= (1ULL << pos); }
static inline void set_bits(bitboard *b, bitboard bits) { *b |= bits; }
static inline void clear_bit(bitboard *b, int pos) { *b &= ~(1ULL << pos); }
static inline void clear_bits(bitboard *b, bitboard bits) { *b &= ~bits; }
static inline bool check_bit(bitboard b, int pos) { return b & (1ULL << pos); }
bitboard apply_masks(bitboard b, int count, ...);

typedef struct board {
    bitboard white_king, black_king;
    bitboard white_pawns, black_pawns;
    bitboard white_knights, black_knights;
    bitboard white_bishops, black_bishops;
    bitboard white_queens, black_queens;
    bitboard white_rooks, black_rooks;
    bitboard white_pieces, black_pieces;
} board;

#define MAX_MOVES 256

typedef struct move_list {
    move moves[MAX_MOVES];
    int count;
} move_list;

move create_move(int from, int to, int flags);
void clear_moves(move_list *l);
bool add_move(move m, move_list *l);
void display_moves(move_list l);

#define STARTING_WHITE_KING bitboard_at(E, 1)
#define STARTING_BLACK_KING bitboard_at(E, 8)

#define STARTING_WHITE_PAWNS bitboard_at(A, 2) | bitboard_at(B, 2) | bitboard_at(C, 2) | \
                             bitboard_at(D, 2) | bitboard_at(E, 2) | bitboard_at(F, 2) | \
                             bitboard_at(G, 2) | bitboard_at(H, 2)

#define STARTING_BLACK_PAWNS bitboard_at(A, 7) | bitboard_at(B, 7) | bitboard_at(C, 7) | \
                             bitboard_at(D, 7) | bitboard_at(E, 7) | bitboard_at(F, 7) | \
                             bitboard_at(G, 7) | bitboard_at(H, 7)

#define STARTING_WHITE_KNIGHTS bitboard_at(B, 1) | bitboard_at(G, 1)
#define STARTING_BLACK_KNIGHTS bitboard_at(B, 8) | bitboard_at(G, 8)

#define STARTING_WHITE_BISHOPS bitboard_at(C, 1) | bitboard_at(F, 1)
#define STARTING_BLACK_BISHOPS bitboard_at(C, 8) | bitboard_at(F, 8)

#define STARTING_WHITE_QUEENS bitboard_at(D, 1)
#define STARTING_BLACK_QUEENS bitboard_at(D, 8)

#define STARTING_WHITE_ROOKS bitboard_at(A, 1) | bitboard_at(H, 1)
#define STARTING_BLACK_ROOKS bitboard_at(A, 8) | bitboard_at(H, 8)

board init_board(void);
board rand_board(void);
bool white_check(board b);
bool black_check(board b);
bool white_checkmate(board b);
bool black_checkmate(board b);
void draw_board(board b);
//TODO piece moves
bool get_white_king_moves(board b, move_list *l);
bool get_white_pawn_moves(board b, move_list *l);
bool get_white_knight_moves(board b, move_list *l);
bool get_white_bishop_moves(board b, move_list *l);
bool get_white_queen_moves(board b, move_list *l);
bool get_white_rook_moves(board b, move_list *l);

bool get_black_king_moves(board b, move_list *l);
bool get_black_pawn_moves(board b, move_list *l);
bool get_black_knight_moves(board b, move_list *l);
bool get_black_bishop_moves(board b, move_list *l);
bool get_black_queen_moves(board b, move_list *l);
bool get_black_rook_moves(board b, move_list *l);

bitboard get_king_attacks(bitboard king);
bitboard get_white_pawn_attacks(bitboard pawns);
bitboard get_black_pawn_attacks(bitboard pawns);
bitboard get_knight_attacks(bitboard knights);
bitboard get_bishop_attacks(bitboard bishops, board b);
bitboard get_rook_attacks(bitboard rooks, board b);
bitboard get_queen_attacks(bitboard queens, board b);

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

int bishop_hash(bitboard blockers, int pos);
int rook_hash(bitboard blockers, int pos);

bitboard get_blockers(bitboard mask, int x);

bitboard get_white_attacks(board b);
bitboard get_black_attacks(board b);

#ifdef __cplusplus
}
#endif

#endif
