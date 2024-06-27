#include "nn.h"
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>

#define WINDOW_X 400
#define WINDOW_Y 100
#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600
#define TILE_WIDTH 30
#define TILE_HEIGHT 30

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

int random_coord(int min, int max) { return rand() % (max - min + 1) + min; }

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

	(*tail) = (*tail)->next;
}

void new_snake(snake_part **head, snake_part **tail)
{
	(*head) = malloc(sizeof(snake_part));

	(*head)->x = random_coord((2 * WINDOW_WIDTH) / (5 * TILE_WIDTH), (WINDOW_WIDTH * 3) / (5 * TILE_WIDTH));
	(*head)->y = random_coord((2 * WINDOW_HEIGHT) / (5 * TILE_HEIGHT), (WINDOW_HEIGHT * 3) / (5 * TILE_HEIGHT));
	(*head)->dir = rand() % 4;
	(*tail) = (*head);
	(*tail)->next = NULL;

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

	*head = NULL;
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
	if (head->x < 0 || head->x > WINDOW_WIDTH / TILE_WIDTH - 1 ||
	    head->y < 0 || head->y > WINDOW_HEIGHT / TILE_HEIGHT - 1) {
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

double gaussian(double x) { return exp(-(pow(x, 2) / 20)); }

void get_inputs(mat inputs, snake_part *head, apple a)
{
	mat_at(inputs, 0, 0) = gaussian(head->x);
	mat_at(inputs, 1, 0) = gaussian(WINDOW_WIDTH / TILE_WIDTH - 1 - head->x);
	mat_at(inputs, 2, 0) = gaussian(WINDOW_HEIGHT / TILE_HEIGHT - 1 - head->y);
	mat_at(inputs, 3, 0) = gaussian(head->y);

	int distance_self_left = 0;
	int distance_self_right = 0;
	int distance_self_down = 0;
	int distance_self_up = 0;
	
	bool has_turned = false;
	snake_part *current = head;
	while (current) {
		if (current->dir != head->dir) has_turned = true;

		if (has_turned && current->y == head->y) {
			if (current->x < head->x) {
				distance_self_left = fmin(distance_self_left, head->x - current->x);
			}
			else if (current->x > head->x) {
				distance_self_right = fmin(distance_self_right, current->x - head->x);
			}
		}

		if (has_turned && current->x == head->x) {
			if (current->y > head->y) {
				distance_self_down = fmin(distance_self_down, current->y - head->y);
			}
			else if (current->y < head->y) {
				distance_self_up = fmin(distance_self_up, head->y - current->y);
			}
		}

		current = current->next;
	}

	mat_at(inputs, 4, 0) = distance_self_left > 0 ? gaussian(distance_self_left) : 0;
	mat_at(inputs, 5, 0) = distance_self_right > 0 ? gaussian(distance_self_right) : 0;
	mat_at(inputs, 6, 0) = distance_self_down > 0 ? gaussian(distance_self_down) : 0;
	mat_at(inputs, 7, 0) = distance_self_up > 0 ? gaussian(distance_self_up) : 0;

	int distance_apple_left = 0;
	int distance_apple_right = 0;
	int distance_apple_down = 0;
	int distance_apple_up = 0;

	if (head->y == a.y) {
		if (a.x < head->x) {
			distance_apple_left = head->x - a.x;
		}
		else if (a.x > head->x) {
			distance_apple_right = a.x - head->x;
		}
	}

	if (head->x == a.x) {
		if (a.y > head->y) {
			distance_apple_down = head->y - a.y;
		}
		else if (a.y < head->y) {
			distance_apple_up = a.y - head->y;
		}
	}

	mat_at(inputs, 8, 0) = distance_apple_left > 0 ? gaussian(distance_apple_left) : 0;
	mat_at(inputs, 9, 0) = distance_apple_right > 0 ? gaussian(distance_apple_right) : 0;
	mat_at(inputs, 10, 0) = distance_apple_down > 0 ? gaussian(distance_apple_down) : 0;
	mat_at(inputs, 11, 0) = distance_apple_up > 0 ? gaussian(distance_apple_up) : 0;

	mat_at(inputs, 12, 0) = head->dir == LEFT ? 1 : 0;
	mat_at(inputs, 13, 0) = head->dir == RIGHT ? 1 : 0;
	mat_at(inputs, 14, 0) = head->dir == DOWN ? 1 : 0;
	mat_at(inputs, 15, 0) = head->dir == UP ? 1 : 0;
}

void change_direction(mat inputs, specimen snake, snake_part *head) 
{
	feed_forward(snake.n, inputs, relu);

	int output = 0;
	double max = 0;
	for (int i = 0; i < 4; ++i) {
		if (mat_at(snake.n.acts[snake.n.layers - 2], i, 0) > max) {
			output = i;
			max = mat_at(snake.n.acts[snake.n.layers - 2], i, 0);
		}
	}

	head->dir = output;
}

double fitness(int score, int frames) { return sqrt(frames * pow(2, 2 * score)); }

int main() 
{
	int score = 0;
	int frames = 0;
	int dead_tracker = 0;
	
	srand(time(0));
	snake_part *head = NULL;
	snake_part *tail = head;
	new_snake(&head, &tail);
	apple a;
	new_apple(head, &a, score);
	
	int layers = 3;
	mat topology = mat_alloc(layers, 1);
	mat_at(topology, 0, 0) = 16;
	mat_at(topology, 1, 0) = 8;
	mat_at(topology, 2, 0) = 4;
	mat inputs = mat_alloc(mat_at(topology, 0, 0), 1);
	
	int gens = 0;
	int gen_count = 0;
	int epochs = 400;
	int progress_report = 100;
	int gen_size = 1000;
	int best_size = 200;

	specimen *gen = gen_alloc(gen_size, layers, topology);
	for (int i = 0; i < gen_size; ++i) {
		net_rand(gen[i].n, -5, 5);
	}

	specimen *new = gen_alloc(gen_size, layers, topology);
	specimen *best = gen_alloc(best_size, layers, topology);
	specimen *offspring = gen_alloc(best_size, layers, topology);


	free(topology.vals);
	topology.vals = NULL;

	double mutation_rate = 5e-2;
	double mean = 0;
	double stddev = 1;

	specimen *showcase = NULL;
	
	while (gens != epochs) {
		while (!showcase) {
			for (int i = 0; i < 100; ++i) {
				while (!check_collisions(head) && dead_tracker < 200) {
					++frames;
					++dead_tracker;
					get_inputs(inputs, head, a);
					change_direction(inputs, gen[gen_count], head);	
					move_snake(head);

					if (check_eat(head, a)) {
						++score;
						dead_tracker = 0;
						increase_snake(&tail);
						new_apple(head, &a, score);
					}	
				}
				
				gen[gen_count].fitness += fitness(score, frames);
				score = 0;
				frames = 0;
				dead_tracker = 0;
				destroy_snake(&head);
				head = NULL;
				tail = head;
				new_snake(&head, &tail);
				new_apple(head, &a, score);
			}

			gen[gen_count++].fitness /= 100;

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

				printf("Generation: %d | Snakes: %d | Best Fitness: %.2f | "
				       "Worst Fitness: %.2f | Avg Fitness: %.2f\n", 
					gens + 1, gen_size, best_fitness, worst_fitness, avg_fitness);
					
				find_best(best, gen, best_size, gen_size);
				for (int i = 0; i < gen_size / best_size; ++i) {
					gen_sbx_crossover(offspring, best, best_size);

					for (int j = 0; j < best_size; ++j) {
						net_copy(new[i * best_size + j].n, offspring[j].n);
					}

					shuffle(best, sizeof(specimen), best_size);
				}

				gen_mutate(new, gen_size, mutation_rate, mean, stddev);
				
				if (++gens % progress_report == 0) {
					qsort(gen, gen_size, sizeof(specimen), compare_fitness);
					showcase = &gen[0];
				}	
				
				gen_copy(&gen, new, gen_size);
			}
		}

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			printf("Couldn't initialize SDL %s\n", SDL_GetError());
			return 1;
		}

		SDL_Window *window = SDL_CreateWindow(
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

		SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

		if (!renderer) {
			printf("Couldn't initialize renderer %s\n", SDL_GetError());
			return 1;
		}
		
		SDL_Event event;
		bool quit = false;

		while (!check_collisions(head) && !quit) {
			SDL_SetRenderDrawColor(renderer, 0x22, 0x22, 0x22, 0xFF);
			SDL_RenderClear(renderer);
			
			get_inputs(inputs, head, a);
			change_direction(inputs, *showcase, head);
			move_snake(head);
		
			if (check_eat(head, a)) {
				increase_snake(&tail);
				new_apple(head, &a, score);
			}

			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_ESCAPE) {
					quit = true;
				}
			}

			render_background(renderer);
			render_snake(renderer, head);
			render_apple(renderer, a);

			SDL_RenderPresent(renderer);
			SDL_Delay(120);
		}

		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		destroy_snake(&head);
		head = NULL;
		tail = head;
		new_snake(&head, &tail);
		new_apple(head, &a, score);
		showcase = NULL;
	}

	gen_destroy(&gen, gen_size);
	gen_destroy(&best, best_size);
	gen_destroy(&offspring, best_size);
	gen_destroy(&new, gen_size);
	free(inputs.vals);
	inputs.vals = NULL;
	destroy_snake(&head);
	SDL_Quit();

	return 0;
}
