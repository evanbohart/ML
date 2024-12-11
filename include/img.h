#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <stdbool.h>

#define WINDOW_X 400
#define WINDOW_Y 100
#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600
#define PIXEL_WIDTH 30
#define PIXEL_HEIGHT 30
#define MAX_X WINDOW_WIDTH / PIXEL_WIDTH
#define MAX_Y WINDOW_HEIGHT / PIXEL_HEIGHT

void set_pixel(bool *pixels, int x, int y);

void render_background(SDL_Renderer *renderer);
void render_pixel(SDL_Renderer *renderer, int x, int y);
