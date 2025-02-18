#ifndef CUBE_H
#define CUBE_H

extern "C" {
    #include "nn.h"
}

#include <cstdint>
#include <stack>
using std::stack;
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
        void get_inputs(tens inputs) const;
        void copy(const cube &c);
        bool is_solved(void) const;
        void scramble(int n);
};

struct state {
    cube c;
    double prior;
    int visits;
    double value;
    state *parent;
    state **children;
};

class tree
{
    private:
        state *root;
        void destroy(state *root);
        void select_child(state *&root);
        void traverse(state *&root);
        void backup(state *leaf, double value);
        void expand_state(cnet cn, net policy, state *root);
        double uct(const state *root) const;
        double eval(cnet cn, net value, state *root);
        void generate_solution(stack<move> &solution, state *leaf);
        void train_value(cnet cn, net value, stack<move> &solution, double rate, state *root);
        void train_policy(cnet cn, net policy, stack<move> &solution, double rate, state *root);
 
    public:
        tree(const cube &c);
        ~tree();
        int mcts(cnet cn, net policy, stack<move> &solution, int n);
        int mcts(cnet cn, net value, net policy, stack<move> &solution, int n);
        void train_value(cnet cn, net value, stack<move> &solution, double rate);
        void train_policy(cnet cn, net policy, stack<move> &solution, double rate);
};

uint64_t roll_left(uint64_t x, int bits);
uint64_t roll_right(uint64_t x, int bits);

#endif
