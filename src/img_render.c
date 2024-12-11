#include "img.h"

void render_background(SDL_Renderer *renderer)
{
    SDL_SetRenderDrawColor(renderer, 0x3B, 0x3B, 0x3B, 0xFF);
    SDL_Rect rect;
    rect.w = WINDOW_WIDTH;
    rect.h = WINDOW_HEIGHT;
    rect.x = 0;
    rect.y = 0;
    SDL_RenderFillRect(renderer, &rect);
}

void render_pixel(SDL_Renderer *renderer, int x, int y)
{
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
    SDL_Rect rect;
    rect.w = PIXEL_WIDTH;
    rect.h = PIXEL_HEIGHT;
    rect.x = x * PIXEL_WIDTH;
    rect.y = y * PIXEL_HEIGHT;
    SDL_RenderFillRect(renderer, &rect);
}
