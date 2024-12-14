#include "cube.h"
#include <iostream>
#include <ctime>

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
    mat_at(value_topology, 1, 0) = 30;
    mat_at(value_topology, 2, 0) = 30;
    mat_at(value_topology, 3, 0) = 1;
    net value = net_alloc(value_layers, value_topology);

    mat policy_topology = mat_alloc(policy_layers, 1);
    mat_at(policy_topology, 0, 0) = 48;
    mat_at(policy_topology, 1, 0) = 30;
    mat_at(policy_topology, 2, 0) = 30;
    mat_at(policy_topology, 3, 0) = 12;
    net policy = net_alloc(policy_layers, policy_topology);

    net_rand(value, -5, 5);
    net_rand(policy, -5, 5);

    mat inputs = mat_alloc(48, 1);

    for (int i = 0; i < 10 * 1000; ++i) {
        model::cube c;
        c.scramble(1);
        ai::tree t(c);
        t.mcts(policy, 50);
        t.train_value(value, 0.05);
        t.train_policy(policy, 0.05);
        std::cout << "Epoch: " << i + 1 << "\n";
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
                c.get_inputs(inputs);

                feed_forward(value, inputs, sig);
                std::cout << "Value:\n";
                mat_print(value.acts[value.layers - 2]);

                feed_forward(policy, inputs, relu);
                mat_softmax(policy.acts[policy.layers - 2], policy.acts[policy.layers - 2]);
                std::cout << "Policy:\n";
                mat_print(policy.acts[policy.layers - 2]);

                double max = 0;
                int index = 0;
                for (int i = 0; i < policy.acts[policy.layers - 2].rows; ++i) {
                    if (mat_at(policy.acts[policy.layers - 2], i, 0) > max) {
                        max = mat_at(policy.acts[policy.layers - 2], i, 0);
                        index = i;
                    }
                }

                c.turn((model::move)index);
            }
        }

        SDL_Delay(500);
    }

    free(value_topology.vals);
    free(policy_topology.vals);
    net_destroy(&value);
    net_destroy(&policy);
    free(inputs.vals);
}
