#include "cube.h"
#include "utils.h"
#include <stdio.h>
#include <cstring>
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

void train(net value, net policy, char *path)
{
    net_glorot(value);
    net_he(policy);

    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 10 * 1000; ++j) {
            for (int k = i; k >= 0; --k) {
                cube c;
                c.scramble(k + 1);
                tree t(c);
                bool solved = t.mcts(policy, 1000 * (k + 1));
                t.train_value(value, 0.01);
                t.train_policy(policy, 0.01);
                printf("Scramble Length: %i | Epoch: %i | %s", k + 1, j + 1, (solved ? "Solved\n" : "Not solved\n"));
            }
        }
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        printf("Error writing to file.\n");
        return;
    }

    net_save(value, f);
    net_save(policy, f);
}

void showcase(net value, net policy, char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("Error reading from file.\n");
        return;
    }

    net_load(&value, f);
    net_load(&policy, f);

    cube c;
    mat inputs = mat_alloc(48, 1);
    stack<move> solution;

    bool quit = false;
    while (!quit) {
        int n;
        printf("Enter scramble length.\n");
        scanf("%i", &n);
        getchar();
        c.scramble(n);
        tree t(c);

        if (!t.solve(value, policy, solution, 25 * 1000)) {
            printf("No solution found.\n");
        }
        else {
            printf("Solution found: ");
            while (!solution.empty()) {
                switch (solution.top()) {
                    case L:
                        printf("L");
                        break;
                    case LPRIME:
                        printf("L'");
                        break;
                    case R:
                        printf("R");
                        break;
                    case RPRIME:
                        printf("R'");
                        break;
                    case D:
                        printf("D");
                        break;
                    case DPRIME:
                        printf("D'");
                        break;
                    case U:
                        printf("U");
                        break;
                    case UPRIME:
                        printf("U'");
                        break;
                    case F:
                        printf("F");
                        break;
                    case FPRIME:
                        printf("F'");
                        break;
                    case B:
                        printf("B");
                        break;
                    case BPRIME:
                        printf("B'");
                        break;
                }

                solution.pop();
            }

            printf("\n");
        }

        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("Couldn't initialize SDL %s\n", SDL_GetError());
            return;
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
            return;
        }

        SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

        if (!renderer) {
            printf("Couldn't initialize renderer %s\n", SDL_GetError());
            return;
        }

        bool esc = false;
        SDL_Event event;
        while (!esc) {
            SDL_RenderClear(renderer);
            render_background(renderer);
            c.render(renderer, 200, 200);
            SDL_RenderPresent(renderer);

            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_ESCAPE) {
                    esc = true;
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_l && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(LPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_l) {
                    c.turn(L);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_r && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(RPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_r) {
                    c.turn(R);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_d && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(DPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_d) {
                    c.turn(D);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_u && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(UPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_u) {
                    c.turn(U);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_f && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(FPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_f) {
                    c.turn(F);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_b && SDL_GetModState() & KMOD_CTRL) {
                    c.turn(BPRIME);
                }
                else if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_b) {
                    c.turn(B);
                }
            }
        }

        SDL_DestroyWindow(window);
        SDL_DestroyRenderer(renderer);

        char q;
        printf("Enter q to quit.\n");
        scanf("%c", &q);
        if (q == 'q') quit = true;
        getchar();
    }

    free(inputs.vals);
}

int main(int argc, char **argv)
{
    srand(time(0));
    int value_layers = 4;
    int policy_layers = 4;

    mat value_topology = mat_alloc(value_layers, 1);
    mat_at(value_topology, 0, 0) = 48;
    mat_at(value_topology, 1, 0) = 256;
    mat_at(value_topology, 2, 0) = 256;
    mat_at(value_topology, 3, 0) = 1;
    net value = net_alloc(value_layers, value_topology);
    for (int i = 0; i < value.layers - 1; ++i) {
        value.actfuncs[i] = SIGMOID;
    }

    mat policy_topology = mat_alloc(policy_layers, 1);
    mat_at(policy_topology, 0, 0) = 48;
    mat_at(policy_topology, 1, 0) = 256;
    mat_at(policy_topology, 2, 0) = 256;
    mat_at(policy_topology, 3, 0) = 12;
    net policy = net_alloc(policy_layers, policy_topology);
    for (int i = 0; i < policy.layers - 2; ++i) {
        policy.actfuncs[i] = RELU;
    }
    policy.actfuncs[policy.layers - 2] = SOFTMAX;

    char path[100];

    if (argc != 3) {
        printf("Usage: %s <mode> <arguments>\n", argv[0]);
    }
    else if (strcmp(argv[1], "train") == 0) {
        get_path(path, argv[2]);
        train(value, policy, path);
    }
    else if (strcmp(argv[1], "showcase") == 0) {
        get_path(path, argv[2]);
        showcase(value, policy, path);
    }
    else {
        printf("Usage: %s <mode> <arguments>\n", argv[0]);
    }

    free(value_topology.vals);
    free(policy_topology.vals);
    net_destroy(&value);
    net_destroy(&policy);
}
