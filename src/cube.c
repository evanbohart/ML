#include "cube.h"
#include <assert.h>

cube init_cube(void)
{
    const uint64_t faces[6] = { 0x0000000000000000, 0x0101010101010101, 0x0202020202020202,
                                0x0303030303030303, 0x0404040404040404, 0x0505050505050505 };

    cube c;

    for (int i = 0; i < 6; ++i) {
        c.faces[i].center = i;
        c.faces[i].bitboard = faces[i];
    }

    return c;
}

void turn_cube_L(cube *c)
{
    turn_face_clockwise(c->faces[1]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_L;
    c->faces[0].bitboard |= c->faces[3].bitboard & ~MASK_OUT_L;
    c->faces[3].bitboard &= MASK_OUT_L;
    c->faces[3].bitboard |= roll_right(c->faces[5].bitboard & ~MASK_OUT_R, 32);
    c->faces[5].bitboard &= MASK_OUT_R;
    c->faces[5].bitboard |= roll_right(c->faces[2].bitboard & ~MASK_OUT_L, 32);
    c->faces[2].bitboard &= MASK_OUT_L;
    c->faces[2].bitboard |= temp & ~MASK_OUT_L;
}

void turn_cube_LPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[1]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_L;
    c->faces[0].bitboard |= c->faces[2].bitboard & ~MASK_OUT_L;
    c->faces[2].bitboard &= MASK_OUT_L;
    c->faces[2].bitboard |= roll_right(c->faces[5].bitboard & ~MASK_OUT_R, 32);
    c->faces[5].bitboard &= MASK_OUT_R;
    c->faces[5].bitboard |= roll_right(c->faces[3].bitboard & ~MASK_OUT_L, 32);
    c->faces[3].bitboard &= MASK_OUT_L;
    c->faces[3].bitboard |= temp & ~MASK_OUT_L;
}

void turn_cube_R(cube *c)
{
    turn_face_clockwise(c->faces[4]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_R;
    c->faces[0].bitboard |= c->faces[2].bitboard & ~MASK_OUT_R;
    c->faces[2].bitboard &= MASK_OUT_R;
    c->faces[2].bitboard |= roll_right(c->faces[5].bitboard & ~MASK_OUT_L, 32);
    c->faces[5].bitboard &= MASK_OUT_L;
    c->faces[5].bitboard |= roll_right(c->faces[3].bitboard & ~MASK_OUT_R, 32);
    c->faces[3].bitboard &= MASK_OUT_R;
    c->faces[3].bitboard |= temp & ~MASK_OUT_R;
}

void turn_cube_RPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[4]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_R;
    c->faces[0].bitboard |= c->faces[3].bitboard & ~MASK_OUT_R;
    c->faces[3].bitboard &= MASK_OUT_R;
    c->faces[3].bitboard |= roll_right(c->faces[5].bitboard & ~MASK_OUT_L, 32);
    c->faces[5].bitboard &= MASK_OUT_L;
    c->faces[5].bitboard |= roll_right(c->faces[2].bitboard & ~MASK_OUT_R, 32);
    c->faces[2].bitboard &= MASK_OUT_R;
    c->faces[2].bitboard |= temp & ~MASK_OUT_R;
}

void turn_cube_D(cube *c)
{
    turn_face_clockwise(c->faces[5]);

    uint64_t temp = c->faces[2].bitboard;

    c->faces[2].bitboard &= MASK_OUT_B;
    c->faces[2].bitboard |= roll_left(c->faces[1].bitboard & ~MASK_OUT_L, 16);
    c->faces[1].bitboard &= MASK_OUT_L;
    c->faces[1].bitboard |= roll_left(c->faces[3].bitboard & ~MASK_OUT_T, 16);
    c->faces[3].bitboard &= MASK_OUT_T;
    c->faces[3].bitboard |= roll_left(c->faces[4].bitboard & ~MASK_OUT_R, 16);
    c->faces[4].bitboard &= MASK_OUT_R;
    c->faces[4].bitboard |= roll_left(temp & ~MASK_OUT_B, 16);
}

void turn_cube_DPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[5]);

    uint64_t temp = c->faces[2].bitboard;

    c->faces[2].bitboard &= MASK_OUT_B;
    c->faces[2].bitboard |= roll_right(c->faces[4].bitboard & ~MASK_OUT_R, 16);
    c->faces[4].bitboard &= MASK_OUT_R;
    c->faces[4].bitboard |= roll_right(c->faces[3].bitboard & ~MASK_OUT_T, 16);
    c->faces[3].bitboard &= MASK_OUT_T;
    c->faces[3].bitboard |= roll_right(c->faces[1].bitboard & ~MASK_OUT_L, 16);
    c->faces[1].bitboard &= MASK_OUT_L;
    c->faces[1].bitboard |= roll_right(temp & ~MASK_OUT_B, 16);
}

void turn_cube_U(cube *c)
{
    turn_face_clockwise(c->faces[0]);

    uint64_t temp = c->faces[2].bitboard;

    c->faces[2].bitboard &= MASK_OUT_T;
    c->faces[2].bitboard |= roll_right(c->faces[4].bitboard & ~MASK_OUT_L, 16);
    c->faces[4].bitboard &= MASK_OUT_L;
    c->faces[4].bitboard |= roll_right(c->faces[3].bitboard & ~MASK_OUT_B, 16);
    c->faces[3].bitboard &= MASK_OUT_B;
    c->faces[3].bitboard |= roll_right(c->faces[1].bitboard & ~MASK_OUT_R, 16);
    c->faces[1].bitboard &= MASK_OUT_R;
    c->faces[1].bitboard |= roll_right(temp & ~MASK_OUT_T, 16);
}

void turn_cube_UPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[0]);

    uint64_t temp = c->faces[2].bitboard;

    c->faces[2].bitboard &= MASK_OUT_T;
    c->faces[2].bitboard |= roll_left(c->faces[1].bitboard & ~MASK_OUT_R, 16);
    c->faces[1].bitboard &= MASK_OUT_R;
    c->faces[1].bitboard |= roll_left(c->faces[3].bitboard & ~MASK_OUT_B, 16);
    c->faces[3].bitboard &= MASK_OUT_B;
    c->faces[3].bitboard |= roll_left(c->faces[4].bitboard & ~MASK_OUT_L, 16);
    c->faces[4].bitboard &= MASK_OUT_L;
    c->faces[4].bitboard |= roll_left(temp & ~MASK_OUT_T, 16);
}

void turn_cube_F(cube *c)
{
    turn_face_clockwise(c->faces[2]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_B;
    c->faces[0].bitboard |= c->faces[1].bitboard & ~MASK_OUT_B;
    c->faces[1].bitboard &= MASK_OUT_B;
    c->faces[1].bitboard |= c->faces[5].bitboard & ~MASK_OUT_B;
    c->faces[5].bitboard &= MASK_OUT_B;
    c->faces[5].bitboard |= c->faces[4].bitboard & ~MASK_OUT_B;
    c->faces[4].bitboard &= MASK_OUT_B;
    c->faces[4].bitboard |= temp & ~MASK_OUT_B;
}

void turn_cube_FPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[2]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_B;
    c->faces[0].bitboard |= c->faces[4].bitboard & ~MASK_OUT_B;
    c->faces[4].bitboard &= MASK_OUT_B;
    c->faces[4].bitboard |= c->faces[5].bitboard & ~MASK_OUT_B;
    c->faces[5].bitboard &= MASK_OUT_B;
    c->faces[5].bitboard |= c->faces[1].bitboard & ~MASK_OUT_B;
    c->faces[1].bitboard &= MASK_OUT_B;
    c->faces[1].bitboard |= temp & ~MASK_OUT_B;
}

void turn_cube_B(cube *c)
{
    turn_face_clockwise(c->faces[3]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_T;
    c->faces[0].bitboard |= c->faces[4].bitboard & ~MASK_OUT_T;
    c->faces[4].bitboard &= MASK_OUT_T;
    c->faces[4].bitboard |= c->faces[5].bitboard & ~MASK_OUT_T;
    c->faces[5].bitboard &= MASK_OUT_T;
    c->faces[5].bitboard |= c->faces[1].bitboard & ~MASK_OUT_T;
    c->faces[1].bitboard &= MASK_OUT_T;
    c->faces[1].bitboard |= temp & ~MASK_OUT_T;
}

void turn_cube_BPRIME(cube *c)
{
    turn_face_counterclockwise(c->faces[3]);

    uint64_t temp = c->faces[0].bitboard;

    c->faces[0].bitboard &= MASK_OUT_T;
    c->faces[0].bitboard |= c->faces[1].bitboard & ~MASK_OUT_T;
    c->faces[1].bitboard &= MASK_OUT_T;
    c->faces[1].bitboard |= c->faces[5].bitboard & ~MASK_OUT_T;
    c->faces[5].bitboard &= MASK_OUT_T;
    c->faces[5].bitboard |= c->faces[4].bitboard & ~MASK_OUT_T;
    c->faces[4].bitboard &= MASK_OUT_T;
    c->faces[4].bitboard |= temp & ~MASK_OUT_T;
}

bool is_solved(cube c)
{
    uint64_t solved_faces[5] = { 0x0000000000000000, 0x0101010101010101,
                                 0x0202020202020202, 0x0303030303030303, 0x0404040404040404 };

    for (int i = 0; i < 5; ++i) {
        if (c.faces[i].bitboard != solved_faces[i]) {
            return false;
        }
    }

    return true;
}

void scramble(cube *c, int n)
{
    for (int i = 0; i < n; ++i) {
        switch (rand() % 12) {
            case 0:
                turn_cube_L(c);
                break;
            case 1:
                turn_cube_LPRIME(c);
                break;
            case 2:
                turn_cube_R(c);
                break;
            case 3:
                turn_cube_RPRIME(c);
                break;
            case 4:
                turn_cube_D(c);
                break;
            case 5:
                turn_cube_DPRIME(c);
                break;
            case 6:
                turn_cube_U(c);
                break;
            case 7:
                turn_cube_UPRIME(c);
                break;
            case 8:
                turn_cube_F(c);
                break;
            case 9:
                turn_cube_FPRIME(c);
                break;
            case 10:
                turn_cube_B(c);
                break;
            case 11:
                turn_cube_BPRIME(c);
                break;
        }
    }

    if (is_solved(*c)) scramble(c, n);
}

void set_color(SDL_Renderer *renderer, color c)
{
    switch (c) {
        case WHITE:
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            break;
        case RED:
            SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF);
            break;
        case BLUE:
            SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
            break;
        case GREEN:
            SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0x00, 0xFF);
            break;
        case ORANGE:
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xA5, 0x00, 0xFF);
            break;
        case YELLOW:
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0x00, 0xFF);
            break;
    }
}

void render_face(SDL_Renderer *renderer, face f, int x, int y)
{
    assert(x > 0);
    assert(y > 0);

    int pos[8][2] = {
        {x, y},
        {x + 20, y},
        {x + 40, y},
        {x + 40, y + 20},
        {x + 40, y + 40},
        {x + 20, y + 40},
        {x, y + 40},
        {x, y + 20}
    };

    for (int i = 0; i < 8; ++i) {
        SDL_Rect rect;
        rect.w = 20;
        rect.h = 20;
        rect.x = pos[i][0];
        rect.y = pos[i][1];

        color c = (color)((f.bitboard & (0xFF00000000000000 >> (i * 8))) >> (56 - (i * 8)));
        set_color(renderer, c);
        SDL_RenderFillRect(renderer, &rect);

        SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderDrawRect(renderer, &rect);
    }

    SDL_Rect rect;
    rect.w = 20;
    rect.h = 20;
    rect.x = x + 20;
    rect.y = y + 20;

    set_color(renderer, f.center);
    SDL_RenderFillRect(renderer, &rect);

    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
    SDL_RenderDrawRect(renderer, &rect);
}

void render_cube(SDL_Renderer *renderer, cube c, int x, int y)
{
    assert(x >= 60);
    assert(y >= 60);

    int pos[6][2] = {
        {x, y},
        {x - 60, y},
        {x, y + 60},
        {x, y - 60},
        {x + 60, y},
        {x + 120, y}
    };

    for (int i = 0; i < 6; ++i) {
        render_face(renderer, c.faces[i], pos[i][0], pos[i][1]);
    }
}
