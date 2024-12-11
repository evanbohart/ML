#include "cube.h"
#include <stdio.h>

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
    c.turn(model::F);
    c.turn(model::F);
    c.turn(model::D);
    c.turn(model::R);
    c.turn(model::R);
    c.turn(model::F);
    c.turn(model::F);
    c.turn(model::U);
    c.turn(model::F);
    c.turn(model::F);
    c.turn(model::R);
    c.turn(model::R);
    c.turn(model::D);
    c.turn(model::D);
    c.turn(model::F);
    c.turn(model::F);
    c.turn(model::R);
    c.turn(model::R);
    c.turn(model::U);
    c.turn(model::B);
    c.turn(model::B);
    c.turn(model::L);
    c.turn(model::BPRIME);
    c.turn(model::UPRIME);
    c.turn(model::R);
    c.turn(model::R);
    c.turn(model::F);
    c.turn(model::L);
    c.turn(model::L);
    c.turn(model::D);
    c.turn(model::D);
    c.turn(model::L);
    c.turn(model::UPRIME);
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
        }
    }
}
