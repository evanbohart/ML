#ifndef CUBE_H
#define CUBE_H

#include <stdint.h>
#include <stdbool.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>

#define WINDOW_X 400
#define WINDOW_Y 100
#define WINDOW_W 600
#define WINDOW_H 600

#define MASK_OUT_T 0x000000FFFFFFFFFF
#define MASK_OUT_B 0xFFFFFFFF000000FF
#define MASK_OUT_L 0x00FFFFFFFFFF0000
#define MASK_OUT_R 0xFFFF000000FFFFFF

typedef enum color { WHITE, RED, BLUE,
                     GREEN, ORANGE, YELLOW } color;

typedef struct face {
    color center;
    uint64_t bitboard;
} face;

#define roll_right(x, bits) ((x) >> (bits) | ((x) << (64 - (bits))))
#define roll_left(x, bits) ((x) << (bits) | ((x) >> (64 - (bits))))

#define turn_face_clockwise(f) ((f).bitboard = roll_right((f).bitboard, 16))
#define turn_face_counterclockwise(f) ((f).bitboard = roll_left((f).bitboard, 16))

typedef struct cube {
    face faces[6];
} cube;

cube init_cube(void);

void turn_cube_L(cube *c);
void turn_cube_LPRIME(cube *c);
void turn_cube_R(cube *c);
void turn_cube_RPRIME(cube *c);
void turn_cube_D(cube *c);
void turn_cube_DPRIME(cube *c);
void turn_cube_U(cube *c);
void turn_cube_UPRIME(cube *c);
void turn_cube_F(cube *c);
void turn_cube_FPRIME(cube *c);
void turn_cube_B(cube *c);
void turn_cube_BPRIME(cube *c);

bool is_solved(cube c);
void scramble(cube *c, int n);

void set_color(SDL_Renderer *renderer, color c);
void render_face(SDL_Renderer *renderer, face f, int x, int y);
void render_cube(SDL_Renderer *renderer, cube c, int x, int y);

#endif
