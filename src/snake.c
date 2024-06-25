#include "nn.h"
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_ttf.h>

#define WINDOW_X 400
#define WINDOW_Y 100
#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600
#define TILE_WIDTH 30
#define TILE_HEIGHT 30

typedef enum {
	TRAIN,
	SHOWCASE
} mode;

enum {
	LEFT,
	RIGHT,
	DOWN,
	UP
};

typedef struct snake_part {
	int x;
	int y;
	int dir;
	struct snake_part *next;
} snake_part;

typedef struct apple {
	int x;
	int y;
} apple;

static inline int random_coord(int min, int max) { return rand() % (max - min + 1) + min; }

void increase_snake(snake_part **tail)
{
	(*tail)->next = malloc(sizeof(snake_part));
	assert((*tail)->next != NULL);

	(*tail)->next->next = NULL;
	(*tail)->next->dir = (*tail)->dir;
	
	switch ((*tail)->dir) {
		case LEFT:
			(*tail)->next->x = (*tail)->x + 1;
			(*tail)->next->y = (*tail)->y;
			break;
		case RIGHT:
			(*tail)->next->x = (*tail)->x - 1;
			(*tail)->next->y = (*tail)->y;
			break;
		case DOWN:
			(*tail)->next->x = (*tail)->x;
			(*tail)->next->y = (*tail)->y - 1;
			break;
		case UP:
			(*tail)->next->x = (*tail)->x;
			(*tail)->next->y = (*tail)->y + 1;
			break;
	}

	*tail = (*tail)->next;
}

void new_snake(snake_part **head, snake_part **tail)
{
	assert(*tail == *head);

	(*head)->x = random_coord((2 * WINDOW_WIDTH) / (5 * TILE_WIDTH), (WINDOW_WIDTH * 3) / (5 * TILE_WIDTH));
	(*head)->y = random_coord((2 * WINDOW_HEIGHT) / (5 * TILE_HEIGHT), (WINDOW_HEIGHT * 3) / (5 * TILE_HEIGHT));
	(*head)->dir = rand() % 4;
	(*head)->next = NULL;

	for (int i = 0; i < 3; ++i) {
		increase_snake(tail);
	}
}

void destroy_snake(snake_part **head)
{
	snake_part *current = *head;
	snake_part *next;

	while (current) {
		next = current->next;
		free(current);
		current = next;
	}
}

void move_snake(snake_part *head)
{
	int prev_dir = head->dir;
	snake_part *prev = NULL;
	snake_part *current = head;
	while (current)
	{
		int current_dir = current->dir;

		switch (current_dir) {
			case LEFT:
				current->x -= 1;
				break;
			case RIGHT:
				current->x += 1;
				break;
			case DOWN:
				current->y += 1;
				break;
			case UP:
				current->y -= 1;
				break;	
		}

		if (prev) current->dir = prev_dir;
	
		prev_dir = current_dir;
		prev = current;
		current = current->next;
	}
}

bool check_collisions(snake_part *head)
{
	if (head->x < 0 || head->x >= WINDOW_WIDTH / TILE_WIDTH ||
	    head->y < 0 || head->y >= WINDOW_HEIGHT / TILE_HEIGHT) {
		return true;
	}

	snake_part *current = head->next;
	while (current) {
		if (head->x == current->x && head->y == current->y) {
			return true;
		}

		current = current->next;
	}

	return false;
}

bool check_eat(snake_part *head, apple a)
{
	if (head->x == a.x && head->y == a.y) {
		return true;
	}

	return false;
}

void new_apple(snake_part *head, apple *a, int score)
{
	int snake_xs[score + 4];
	int snake_ys[score + 4];

	snake_part *current = head;
	int i = 0;
	while (current) {
		snake_xs[i] = current->x;
		snake_ys[i] = current->y;
		++i;

		current = current->next;
	}

	bool valid = false;
	while (!valid) {
		a->x = random_coord(0, WINDOW_WIDTH / TILE_WIDTH - 1);
		a->y = random_coord(0, WINDOW_HEIGHT / TILE_HEIGHT - 1);

		valid = true;
		for (int i = 0; i < score; ++i) {
			if (snake_xs[i] == a->x && snake_ys[i] == a->y) {
				valid = false;
			}
		}
	}
}

bool get_game_over(snake_part *head, snake_part **tail, apple *a, int *score)
{
	move_snake(head);

	if (check_collisions(head)) {
		return true;
	}

	if (check_eat(head, *a)) {
		++(*score);
		increase_snake(tail);
		new_apple(head, a, *score);
	}

	return false;
}

void render_background(SDL_Renderer *renderer)
{
	SDL_SetRenderDrawColor(renderer, 0x3B, 0x3B, 0x3B, 0xFF);

	for (int i = 0; i < WINDOW_WIDTH / TILE_WIDTH; ++i) {
		for (int j = 0; j < WINDOW_HEIGHT / TILE_HEIGHT; ++j) {
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

void render_score(SDL_Renderer *renderer, int score)
{
	TTF_Font *arial = TTF_OpenFont("arial.ttf", 24);
	SDL_Color white;
	white.r = 0xFF;
	white.g = 0xFF;
	white.b = 0xFF;

	char text_buffer[32];
	sprintf(text_buffer, "Score: %d", score);
	SDL_Surface *text_surface = TTF_RenderText_Solid(arial, text_buffer, white); 
	SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, text_surface);

	SDL_Rect rect;
	rect.x = 400;
	rect.y = 400;
	rect.w = 100;
	rect.h = 100;
	SDL_RenderCopy(renderer, text, NULL, &rect);

	TTF_CloseFont(arial);
	SDL_FreeSurface(text_surface);
	SDL_DestroyTexture(text);
}

void render_time(SDL_Renderer *renderer, int time) {
	TTF_Font *arial = TTF_OpenFont("arial.ttf", 24);
	SDL_Color white;
	white.r = 0xFF;
	white.g = 0xFF;
	white.b = 0xFF;

	int minutes = time / 60;
	int seconds = time - minutes * 60;
	char text_buffer[32];
	sprintf(text_buffer, "Time: %02d:%02d", minutes, seconds);
	SDL_Surface *text_surface = TTF_RenderText_Solid(arial, text_buffer, white); 
	SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, text_surface);

	SDL_Rect rect;
	rect.x = 100;
	rect.y = 100;
	rect.w = 100;
	rect.h = 100;
	SDL_RenderCopy(renderer, text, NULL, &rect);

	TTF_CloseFont(arial);
	SDL_FreeSurface(text_surface);
	SDL_DestroyTexture(text);
}

void get_inputs(mat inputs, snake_part *head, snake_part *tail, apple a)
{
	assert(inputs.rows == 24);
	assert(inputs.cols == 1);

	mat_at(inputs, 0, 0) = gaussian(head->x);
	mat_at(inputs, 1, 0) = gaussian((WINDOW_WIDTH / TILE_WIDTH - 1) - head->x);
	mat_at(inputs, 2, 0) = gaussian((WINDOW_HEIGHT / TILE_HEIGHT - 1) - head->y);
	mat_at(inputs, 3, 0) = gaussian(head->y);
	mat_at(inputs, 4, 0) = gaussian(tail->x);
	mat_at(inputs, 5, 0) = gaussian((WINDOW_WIDTH / TILE_WIDTH - 1) - tail->x);
	mat_at(inputs, 6, 0) = gaussian((WINDOW_HEIGHT / TILE_HEIGHT - 1) - tail->y);
	mat_at(inputs, 7, 0) = gaussian(tail->y);

	int head_distance_left = WINDOW_WIDTH / TILE_WIDTH - 1;
	int head_distance_right = WINDOW_WIDTH / TILE_WIDTH - 1;
	int head_distance_up = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	int head_distance_down = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	int tail_distance_left = WINDOW_WIDTH / TILE_WIDTH - 1;
	int tail_distance_right = WINDOW_WIDTH / TILE_WIDTH - 1;
	int tail_distance_up = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	int tail_distance_down = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	
	snake_part *current = head;
	while (current) {
		if (current->y == head->y) {
			if (current->x < head->x) {
				head_distance_left = fmin(head_distance_left, head->x - current->x);
			}
			else if (current->x > head->x) {
				head_distance_right = fmin(head_distance_right, current->x - head->x);
			}
		}

		if (current->x == head->x) {
			if (current->y > head->y) {
				head_distance_down = fmin(head_distance_down, current->y - head->y);
			}
			else if (current->y < head->y) {
				head_distance_up = fmin(head_distance_up, head->y - current->y);
			}
		}

		if (current->y == tail->y) {
			if (current->x < tail->x) {
				tail_distance_left = fmin(tail_distance_left, tail->x - current->x);
			}
			else if (current->x > tail->x) {
				tail_distance_right = fmin(tail_distance_right, current->x - tail->x);
			}
		}

		if (current->x == tail->x) {
			if (current->y > tail->y) {
				tail_distance_down = fmin(tail_distance_down, current->y - tail->y);
			}
			else if (current->y < tail->y) {
				tail_distance_up = fmin(tail_distance_up, tail->y - current->y);
			}
		}

		current = current->next;
	}

	mat_at(inputs, 8, 0) = gaussian(head_distance_left);
	mat_at(inputs, 9, 0) = gaussian(head_distance_right);
	mat_at(inputs, 10, 0) = gaussian(head_distance_down);
	mat_at(inputs, 11, 0) = gaussian(head_distance_up);
	mat_at(inputs, 12, 0) = gaussian(tail_distance_left);
	mat_at(inputs, 13, 0) = gaussian(tail_distance_right);
	mat_at(inputs, 14, 0) = gaussian(tail_distance_down);
	mat_at(inputs, 15, 0) = gaussian(tail_distance_up);

	head_distance_left = WINDOW_WIDTH / TILE_WIDTH - 1;
	head_distance_right = WINDOW_WIDTH / TILE_WIDTH - 1;
	head_distance_up = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	head_distance_down = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	tail_distance_left = WINDOW_WIDTH / TILE_WIDTH - 1;
	tail_distance_right = WINDOW_WIDTH / TILE_WIDTH - 1;
	tail_distance_up = WINDOW_HEIGHT / TILE_HEIGHT - 1;
	tail_distance_down = WINDOW_HEIGHT / TILE_HEIGHT - 1;

	if (head->x == a.x) {
		if (head->y < a.y) {
			head_distance_left = a.y - head->y;
		}
		else if (head->y > a.y) {
			head_distance_right = head->y - a.y;
		}
	}

	if (head->y == a.y) {
		if (head->y > a.y) {
			head_distance_down = head->y - a.y;
		}
		else if (head->y < a.y) {
			head_distance_up = a.y - head->y;
		}
	}

	if (tail->x == a.x) {
		if (tail->y < a.y) {
			tail_distance_left = a.y - tail->y;
		}
		else if (tail->y > a.y) {
			tail_distance_right = tail->y - a.y;
		}
	}

	if (tail->y == a.y) {
		if (tail->y > a.y) {
			tail_distance_down = tail->y - a.y;
		}
		else if (tail->y < a.y) {
			tail_distance_up = a.y - tail->y;
		}
	}

	mat_at(inputs, 16, 0) = gaussian(head_distance_left);
	mat_at(inputs, 17, 0) = gaussian(head_distance_right);
	mat_at(inputs, 18, 0) = gaussian(head_distance_down);
	mat_at(inputs, 19, 0) = gaussian(head_distance_up);
	mat_at(inputs, 20, 0) = gaussian(tail_distance_left);
	mat_at(inputs, 21, 0) = gaussian(tail_distance_right);
	mat_at(inputs, 22, 0) = gaussian(tail_distance_down);
	mat_at(inputs, 23, 0) = gaussian(tail_distance_up);
}

void change_direction(mat inputs, specimen snake, snake_part **head) 
{
	feed_forward(snake.n, inputs, sig);

	int output;
	double max = 0;
	for (int i = 0; i < mat_at(snake.n.topology, snake.n.layers - 1, 0); ++i) {
		if (mat_at(snake.n.acts[snake.n.layers - 2], i, 0) > max) {
			output = i;	
			max = mat_at(snake.n.acts[snake.n.layers - 2], i, 0);
		}
	}

	switch (output) {
		case LEFT:
			(*head)->dir = (*head)->dir == RIGHT ? RIGHT : LEFT;
			break;
		case RIGHT:
			(*head)->dir = (*head)->dir == LEFT ? LEFT : RIGHT;
			break;
		case DOWN:
			(*head)->dir = (*head)->dir == UP ? UP : DOWN;
			break;
		case UP:
			(*head)->dir = (*head)->dir == DOWN ? DOWN : UP;
			break;
	}
}

static inline double fitness(int score, double time) { return score + time / 60; }

int main() 
{		
	int score = 0;
	double timer = 0;

	srand(time(0));
	snake_part *head = malloc(sizeof(snake_part));
	snake_part *tail = head;
	new_snake(&head, &tail);
	apple a;
	new_apple(head, &a, score);
	
	int layers = 4;
	mat topology = mat_alloc(layers, 1);
	mat_at(topology, 0, 0) = 24;
	mat_at(topology, 1, 0) = 16;
	mat_at(topology, 2, 0) = 16;
	mat_at(topology, 3, 0) = 4;
	mat inputs = mat_alloc(mat_at(topology, 0, 0), 1);
	
	int gens = 0;
	int gen_count = 0;
	int epochs = 1000;
	int gen_size = 1000;
	int best_size = 20;
	int offspring_size = best_size / 2;

	specimen *gen = gen_alloc(gen_size, layers, topology);
	for (int i = 0; i < gen_size; ++i) {
		net_rand(gen[i].n, -10, 10);
	}

	specimen *new = gen_alloc(gen_size, layers, topology);
	specimen *best = gen_alloc(best_size, layers, topology);
	specimen *offspring = gen_alloc(offspring_size, layers, topology);

	specimen showcase;
	mode m = TRAIN;

	while (m == TRAIN) {
		while (!get_game_over(head, &tail, &a, &score)) {
			get_inputs(inputs, head, tail, a);
			change_direction(inputs, gen[gen_count], &head);
			timer += 120.0 / 1000.0;
		}

		gen[gen_count++].fitness = fitness(score, timer);
		score = 0;
		timer = 0;
		destroy_snake(&head->next);
		tail = head;
		new_snake(&head, &tail);
		new_apple(head, &a, score);

		if (gen_count == gen_size) {
			gen_count = 0;

			double best_fitness = gen[0].fitness;
			double worst_fitness = gen[0].fitness;
			double avg_fitness = 0.0;
			for (int i = 0; i < gen_size; ++i) {
				best_fitness = fmax(best_fitness, gen[i].fitness);
				worst_fitness = fmin(worst_fitness, gen[i].fitness);
				avg_fitness += gen[i].fitness;
			}
			avg_fitness /= gen_size;

			printf("Generation: %d | Snakes: %d | Best Fitness: %f | "
			       "Worst Fitness: %f | Avg Fitness: %f\n", 
				gens + 1, gen_size, best_fitness, worst_fitness, avg_fitness);

			find_best(best, gen, best_size, gen_size);

			for (int i = 0; i < gen_size / offspring_size; ++i) {
				gen_breed(offspring, best, offspring_size, best_size);

				for (int j = 0; j < offspring_size; ++j) {
					net_copy(new[i * offspring_size + j].n, offspring[j].n);
				}
			}

			gen_mutate(new, gen_size, -.5, .5);
			
			if (++gens == epochs) {
				showcase = gen[0];
				gen_destroy(&gen, gen_size);
				gen_destroy(&best, best_size);
				gen_destroy(&offspring, offspring_size);
				gen_destroy(&new, gen_size);

				free(topology.vals);
				topology.vals = NULL;
				free(inputs.vals);
				inputs.vals = NULL;	

				m = SHOWCASE;
			}
			else {
				gen_copy(&gen, new, gen_size);
			}
		}
	}

	SDL_Window *window;
	SDL_Renderer *renderer;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("Couldn't initialize SDL %s\n", SDL_GetError());
		return 1;
	}

	if (TTF_Init() < 0) {
		printf("Couldn't initialize TTF %s\n", TTF_GetError());
		return 1;
	}

	window = SDL_CreateWindow(
		"Snake",
		WINDOW_X,
		WINDOW_Y,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		SDL_WINDOW_BORDERLESS
	);

	if (!window) {
		printf("Couldn't open window %s\n", SDL_GetError());
		return 1;
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	if (!renderer) {
		printf("Couldn't initialize renderer %s\n", SDL_GetError());
		return 1;
	}
	
	bool quit = false;
	SDL_Event event;

	while (!quit) {
		SDL_SetRenderDrawColor(renderer, 0x22, 0x22, 0x22, 0xFF);
		SDL_RenderClear(renderer);
		
		get_inputs(inputs, head, tail, a);
		change_direction(inputs, showcase, &head);

		if (get_game_over(head, &tail, &a, &score)) {
			score = 0;
			timer = 0;
			destroy_snake(&head->next);
			tail = head;
			new_snake(&head, &tail);
			new_apple(head, &a, score);
		}
		
		render_background(renderer);
		render_snake(renderer, head);
		render_apple(renderer, a);
		render_score(renderer, score);
		render_time(renderer, timer);

		SDL_RenderPresent(renderer);

		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				quit = true;
			}
			
			if (SDL_GetKeyboardState(NULL)[SDL_SCANCODE_ESCAPE]) {
				quit = true;
			}
		}	

		SDL_Delay(120);
		timer += 120.0 / 1000.0;
	}

	destroy_snake(&head);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
