#include "cube.h"
#include <iostream>
#include <ctime>
#include <cmath>

void render_background(SDL_Renderer *renderer)
{
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
    SDL_Rect rect;
    rect.w = WINDOW_W;
    rect.h = WINDOW_H;
    rect.x = 0;
    rect.y = 0;
    SDL_RenderFillRect(renderer, &rect);
}

int main(void)
{
    srand(time(0));
    int value_layers = 4;
    int policy_layers = 4;

    mat value_topology = mat_alloc(value_layers, 1);
    mat_at(value_topology, 0, 0) = 48;
    mat_at(value_topology, 1, 0) = 60;
    mat_at(value_topology, 2, 0) = 60;
    mat_at(value_topology, 3, 0) = 1;
    net value = net_alloc(value_layers, value_topology);
    for (int i = 0; i < value.layers - 1; ++i) {
        value.actfuncs[i] = SIGMOID;
    }

    mat policy_topology = mat_alloc(policy_layers, 1);
    mat_at(policy_topology, 0, 0) = 48;
    mat_at(policy_topology, 1, 0) = 60;
    mat_at(policy_topology, 2, 0) = 60;
    mat_at(policy_topology, 3, 0) = 12;
    net policy = net_alloc(policy_layers, policy_topology);
    for (int i = 0; i < policy.layers - 2; ++i) {
        policy.actfuncs[i] = RELU;
    }
    policy.actfuncs[policy.layers - 2] = SOFTMAX;

    net_glorot(value);
    net_he(policy);

    mat inputs = mat_alloc(48, 1);

    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1 * 1000; ++j) {
            model::cube c;
            c.scramble(i + 1);
            ai::tree t(c);
            t.mcts(policy, 100);
            t.train_value(value, 0.01);
            t.train_policy(policy, 0.01);
            std::cout << "Scramble Length: " << i + 1 << " | Epoch: " << j + 1 << "\n";
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Couldn't initialize SDL %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(
        "Cube",
        WINDOW_X,
        WINDOW_Y,
        WINDOW_W,
        WINDOW_H,
        SDL_WINDOW_BORDERLESS
    );

    if (!window) {
        printf("Couldn't initialize window %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (!renderer) {
        printf("Couldn't initialize renderer %s\n", SDL_GetError());
        return 1;
    }

    model::cube c;
    c.scramble(1);
    ai::tree t(c);
    c.get_inputs(inputs);
    feed_forward(policy, inputs);
    mat_print(policy.acts[value.layers - 2]);
    stack<model::move> solution;

    if (!t.solve(value, policy, solution, 12000)) {
        std::cout << "No solution found.\n";
    }
    bool quit = false;
    SDL_Event event;

    while (!quit) {
        SDL_RenderClear(renderer);
        render_background(renderer);
        c.render(renderer, 200, 200);
        SDL_RenderPresent(renderer);

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_ESCAPE) {
                quit = true;
            }

            if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_RETURN) {
                c.turn(solution.top());
                solution.pop();
            }
        }
    }

    free(value_topology.vals);
    free(policy_topology.vals);
    net_destroy(&value);
    net_destroy(&policy);
    free(inputs.vals);
}
