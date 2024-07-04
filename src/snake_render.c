#include "snake.h"

void render_background(SDL_Renderer *renderer)
{
	SDL_SetRenderDrawColor(renderer, 0x3B, 0x3B, 0x3B, 0xFF);

	for (int i = 0; i < MAX_X; ++i) {
		for (int j = 0; j < MAX_Y; ++j) {
			SDL_Rect rect;
			rect.w = TILE_WIDTH;
			rect.h = TILE_HEIGHT;
			rect.x = i * TILE_WIDTH;
			rect.y = j * TILE_HEIGHT;

			SDL_RenderDrawRect(renderer, &rect);
		}
	}
}

void render_snake(SDL_Renderer *renderer, snake_part *head)
{
	Uint8 r = 0x19;
	Uint8 g = 0x84;
	Uint8 b = 0x50;

	snake_part *current = head;
	while (current) {
		SDL_SetRenderDrawColor(renderer, r, g, b, 0xFF);
		SDL_Rect rect;
		rect.w = TILE_WIDTH;
		rect.h = TILE_HEIGHT;
		rect.x = current->x * TILE_WIDTH;
		rect.y = current->y * TILE_HEIGHT;
		SDL_RenderFillRect(renderer, &rect);
		current = current->next;
		
		if (g < 0xCC) {
			r += 0x05;
			g += 0x05;
			b += 0x05;
		}
	}
}

void render_apple(SDL_Renderer *renderer, apple a)
{
	SDL_SetRenderDrawColor(renderer, 0xA9, 0x1B, 0x0D, 0xFF);
	SDL_Rect rect;
	rect.w = TILE_WIDTH;
	rect.h = TILE_HEIGHT;
	rect.x = a.x * TILE_WIDTH;
	rect.y = a.y * TILE_HEIGHT;
	SDL_RenderFillRect(renderer, &rect);
}
