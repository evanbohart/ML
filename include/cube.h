#ifndef CUBE_H
#define CUBE_H

#include <cstdint>
#define SDL_MAIN_HANDLED
#include <SDL.h>

const int WINDOW_X = 400;
const int WINDOW_Y = 100;
const int WINDOW_W = 600;
const int WINDOW_H = 600;

const uint64_t MASK_OUT_T = 0x000000FFFFFFFFFF;
const uint64_t MASK_OUT_B = 0xFFFFFFFF000000FF;
const uint64_t MASK_OUT_L = 0x00FFFFFFFFFF0000;
const uint64_t MASK_OUT_R = 0xFFFF000000FFFFFF;

namespace model
{
    enum color : uint8_t
    {
        WHITE,
        RED,
        BLUE,
        GREEN,
        ORANGE,
        YELLOW
    };

    enum move : uint8_t
    {
        L, LPRIME,
        R, RPRIME,
        D, DPRIME,
        U, UPRIME,
        F, FPRIME,
        B, BPRIME
    };

    struct face
    {
        color center; 
        uint64_t bitboard;
    };

    class cube
    {
        private:
            face faces[6];
            void turn_face_clockwise(face &f);
            void turn_face_counterclockwise(face &f);
            void set_color(SDL_Renderer *renderer, color c);
            void render_face(SDL_Renderer *renderer, const face &f, int x, int y);
        public:
            cube();
            void render(SDL_Renderer *renderer, int x, int y);
            void turn(const move &m);
    };
}

namespace utils
{
    uint64_t roll_left(uint64_t x, int bits);
    uint64_t roll_right(uint64_t x, int bits);
}

#endif
