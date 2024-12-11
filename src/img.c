#include "nn.h"
#include "img.h"

int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Couldn't initialize SDL, %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(
        "title",
        WINDOW_X,
        WINDOW_Y,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
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
		
    SDL_Event event;

    int mouse_x = 0;
    int mouse_y = 0;

    bool pixels[MAX_X * MAX_Y];
    for (int i = 0; i < MAX_X * MAX_Y; ++i) {
        pixels[i] = false;
    }

    bool drawing = false;
    bool quit = false;

    while (!quit) {
        SDL_SetRenderDrawColor(renderer, 0x3B, 0x3B, 0x3B, 0xFF);
        SDL_RenderClear(renderer);

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_ESCAPE) {
                quit = true;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
				drawing = true;
			}
			else if (event.type == SDL_MOUSEBUTTONUP) {
				drawing = false;
			}
			else if (event.type == SDL_MOUSEMOTION) {
				mouse_x = event.motion.x / PIXEL_WIDTH;
				mouse_y = event.motion.y / PIXEL_HEIGHT;
			}
        }

        if (drawing) {
            set_pixel(pixels, mouse_x, mouse_y);
        }
 
        for (int i = 0; i < MAX_Y; ++i) {
            for (int j = 0; j < MAX_X; ++j) {
                 if (pixels[i * MAX_X + j]) render_pixel(renderer, j, i);
            }
        }

        SDL_RenderPresent(renderer);

    }
    
    int layers = 4;
    mat topology = mat_alloc(layers, 1);
    mat_at(topology, 0, 0) = 
    net n = net = 
}
