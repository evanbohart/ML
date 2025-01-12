#include "cube.h"
#include <cassert>
#include <cstdlib>
#include <utility>
using std::pair;

cube::cube()
{
    for (int i = 0; i < 6; ++i) {
        faces[i].center = (color)i;
        faces[i].bitboard = 0;

        for (int j = 0; j < 8; ++j) {
            faces[i].bitboard |= ((uint64_t)i << (j * 8));
        }
    }
}

void cube::turn_face_clockwise(face &f)
{
    f.bitboard = roll_right(f.bitboard, 16);
}

void cube::turn_face_counterclockwise(face &f)
{
    f.bitboard = roll_left(f.bitboard, 16);
}

void cube::turn(const move &m)
{
    uint64_t temp;

    switch (m) {
        case L:
            turn_face_clockwise(faces[1]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_L;
            faces[0].bitboard |= faces[3].bitboard & ~MASK_OUT_L;
            faces[3].bitboard &= MASK_OUT_L;
            faces[3].bitboard |= roll_right(faces[5].bitboard & ~MASK_OUT_R, 32);
            faces[5].bitboard &= MASK_OUT_R;
            faces[5].bitboard |= roll_right(faces[2].bitboard & ~MASK_OUT_L, 32);
            faces[2].bitboard &= MASK_OUT_L;
            faces[2].bitboard |= temp & ~MASK_OUT_L;

            break;
        case LPRIME:
            turn_face_counterclockwise(faces[1]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_L;
            faces[0].bitboard |= faces[2].bitboard & ~MASK_OUT_L;
            faces[2].bitboard &= MASK_OUT_L;
            faces[2].bitboard |= roll_right(faces[5].bitboard & ~MASK_OUT_R, 32);
            faces[5].bitboard &= MASK_OUT_R;
            faces[5].bitboard |= roll_right(faces[3].bitboard & ~MASK_OUT_L, 32);
            faces[3].bitboard &= MASK_OUT_L;
            faces[3].bitboard |= temp & ~MASK_OUT_L;

            break;
        case R:
            turn_face_clockwise(faces[4]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_R;
            faces[0].bitboard |= faces[2].bitboard & ~MASK_OUT_R;
            faces[2].bitboard &= MASK_OUT_R;
            faces[2].bitboard |= roll_right(faces[5].bitboard & ~MASK_OUT_L, 32);
            faces[5].bitboard &= MASK_OUT_L;
            faces[5].bitboard |= roll_right(faces[3].bitboard & ~MASK_OUT_R, 32);
            faces[3].bitboard &= MASK_OUT_R;
            faces[3].bitboard |= temp & ~MASK_OUT_R;

            break;
        case RPRIME:
            turn_face_counterclockwise(faces[4]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_R;
            faces[0].bitboard |= faces[3].bitboard & ~MASK_OUT_R;
            faces[3].bitboard &= MASK_OUT_R;
            faces[3].bitboard |= roll_right(faces[5].bitboard & ~MASK_OUT_L, 32);
            faces[5].bitboard &= MASK_OUT_L;
            faces[5].bitboard |= roll_right(faces[2].bitboard & ~MASK_OUT_R, 32);
            faces[2].bitboard &= MASK_OUT_R;
            faces[2].bitboard |= temp & ~MASK_OUT_R;

            break;
        case D:
            turn_face_clockwise(faces[5]);

            temp = faces[2].bitboard;

            faces[2].bitboard &= MASK_OUT_B;
            faces[2].bitboard |= roll_left(faces[1].bitboard & ~MASK_OUT_L, 16);
            faces[1].bitboard &= MASK_OUT_L;
            faces[1].bitboard |= roll_left(faces[3].bitboard & ~MASK_OUT_T, 16);
            faces[3].bitboard &= MASK_OUT_T;
            faces[3].bitboard |= roll_left(faces[4].bitboard & ~MASK_OUT_R, 16);
            faces[4].bitboard &= MASK_OUT_R;
            faces[4].bitboard |= roll_left(temp & ~MASK_OUT_B, 16);

            break;
        case DPRIME:
            turn_face_counterclockwise(faces[5]);

            temp = faces[2].bitboard;

            faces[2].bitboard &= MASK_OUT_B;
            faces[2].bitboard |= roll_right(faces[4].bitboard & ~MASK_OUT_R, 16);
            faces[4].bitboard &= MASK_OUT_R;
            faces[4].bitboard |= roll_right(faces[3].bitboard & ~MASK_OUT_T, 16);
            faces[3].bitboard &= MASK_OUT_T;
            faces[3].bitboard |= roll_right(faces[1].bitboard & ~MASK_OUT_L, 16);
            faces[1].bitboard &= MASK_OUT_L;
            faces[1].bitboard |= roll_right(temp & ~MASK_OUT_B, 16);

            break;
        case U:
            turn_face_clockwise(faces[0]);

            temp = faces[2].bitboard;

            faces[2].bitboard &= MASK_OUT_T;
            faces[2].bitboard |= roll_right(faces[4].bitboard & ~MASK_OUT_L, 16);
            faces[4].bitboard &= MASK_OUT_L;
            faces[4].bitboard |= roll_right(faces[3].bitboard & ~MASK_OUT_B, 16);
            faces[3].bitboard &= MASK_OUT_B;
            faces[3].bitboard |= roll_right(faces[1].bitboard & ~MASK_OUT_R, 16);
            faces[1].bitboard &= MASK_OUT_R;
            faces[1].bitboard |= roll_right(temp & ~MASK_OUT_T, 16);

            break;
        case UPRIME:
            turn_face_counterclockwise(faces[0]);

            temp = faces[2].bitboard;

            faces[2].bitboard &= MASK_OUT_T;
            faces[2].bitboard |= roll_left(faces[1].bitboard & ~MASK_OUT_R, 16);
            faces[1].bitboard &= MASK_OUT_R;
            faces[1].bitboard |= roll_left(faces[3].bitboard & ~MASK_OUT_B, 16);
            faces[3].bitboard &= MASK_OUT_B;
            faces[3].bitboard |= roll_left(faces[4].bitboard & ~MASK_OUT_L, 16);
            faces[4].bitboard &= MASK_OUT_L;
            faces[4].bitboard |= roll_left(temp & ~MASK_OUT_T, 16);

            break;
        case F:
            turn_face_clockwise(faces[2]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_B;
            faces[0].bitboard |= faces[1].bitboard & ~MASK_OUT_B;
            faces[1].bitboard &= MASK_OUT_B;
            faces[1].bitboard |= faces[5].bitboard & ~MASK_OUT_B;
            faces[5].bitboard &= MASK_OUT_B;
            faces[5].bitboard |= faces[4].bitboard & ~MASK_OUT_B;
            faces[4].bitboard &= MASK_OUT_B;
            faces[4].bitboard |= temp & ~MASK_OUT_B;

            break;
        case FPRIME:
            turn_face_counterclockwise(faces[2]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_B;
            faces[0].bitboard |= faces[4].bitboard & ~MASK_OUT_B;
            faces[4].bitboard &= MASK_OUT_B;
            faces[4].bitboard |= faces[5].bitboard & ~MASK_OUT_B;
            faces[5].bitboard &= MASK_OUT_B;
            faces[5].bitboard |= faces[1].bitboard & ~MASK_OUT_B;
            faces[1].bitboard &= MASK_OUT_B;
            faces[1].bitboard |= temp & ~MASK_OUT_B;

            break;
        case B:
            turn_face_clockwise(faces[3]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_T;
            faces[0].bitboard |= faces[4].bitboard & ~MASK_OUT_T;
            faces[4].bitboard &= MASK_OUT_T;
            faces[4].bitboard |= faces[5].bitboard & ~MASK_OUT_T;
            faces[5].bitboard &= MASK_OUT_T;
            faces[5].bitboard |= faces[1].bitboard & ~MASK_OUT_T;
            faces[1].bitboard &= MASK_OUT_T;
            faces[1].bitboard |= temp & ~MASK_OUT_T;

            break;
        case BPRIME:
            turn_face_counterclockwise(faces[3]);

            temp = faces[0].bitboard;

            faces[0].bitboard &= MASK_OUT_T;
            faces[0].bitboard |= faces[1].bitboard & ~MASK_OUT_T;
            faces[1].bitboard &= MASK_OUT_T;
            faces[1].bitboard |= faces[5].bitboard & ~MASK_OUT_T;
            faces[5].bitboard &= MASK_OUT_T;
            faces[5].bitboard |= faces[4].bitboard & ~MASK_OUT_T;
            faces[4].bitboard &= MASK_OUT_T;
            faces[4].bitboard |= temp & ~MASK_OUT_T;

            break;
    }
}

bool cube::is_solved(void) const
{
    uint64_t solved_faces[5] = {0x0000000000000000, 0x0101010101010101,
                                0x0202020202020202, 0x0303030303030303,
                                0x0404040404040404};

    for (int i = 0; i < 5; ++i) {
        if (faces[i].bitboard != solved_faces[i]) {
            return false;
        }
    }

    return true;
}

void cube::get_inputs(tens inputs) const
{
    assert(inputs.rows == 8);
    assert(inputs.cols == 6);
    assert(inputs.depth == 1);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 6; ++j) {
            tens_at(inputs, i, j, 0) = (faces[j].bitboard & (0xFF00000000000000 >> (i * 8))) >> (56 - (i * 8));
        }
    }
}

void cube::copy(const cube &c)
{
    for (int i = 0; i < 6; ++i) {
        faces[i].bitboard = c.faces[i].bitboard;
    }
}

void cube::scramble(int n)
{
    for (int i = 0; i < n; ++i) {
        turn((move)(rand() % 12));
    }

    if (is_solved()) scramble(n);
}

void cube::set_color(SDL_Renderer *renderer, color c)
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

void cube::render_face(SDL_Renderer *renderer, const face &f, int x, int y)
{
    assert(x > 0);
    assert(y > 0);

    const pair<int, int> pos[8] = {
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
        rect.x = pos[i].first;
        rect.y = pos[i].second;

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

void cube::render(SDL_Renderer *renderer, int x, int y)
{
    assert(x >= 60);
    assert(y >= 60);

    const pair<int, int> pos[6] = {
        {x, y},
        {x - 60, y},
        {x, y + 60},
        {x, y - 60},
        {x + 60, y},
        {x + 120, y}
    };

    for (int i = 0; i < 6; ++i) {
        render_face(renderer, faces[i], pos[i].first, pos[i].second);
    }
}
