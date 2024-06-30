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
#define MAX_X WINDOW_WIDTH / TILE_WIDTH
#define MAX_Y WINDOW_HEIGHT / TILE_HEIGHT

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

int random_coord(int min, int max) { return rand() % (max - min) + min; }

void increase_snake(snake_part **tail)
{
	assert(*tail);
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
	assert(!*head);
	assert(!*tail);
	*head = malloc(sizeof(snake_part));

	(*head)->x = random_coord(2 * MAX_X / 5, 3 * MAX_X / 5);
	(*head)->y = random_coord(2 * MAX_Y / 5, 3 * MAX_Y / 5);
	(*head)->dir = rand() % 4;
	*tail = *head;
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
	if (head->x < 0 || head->x >= MAX_X ||
	    head->y < 0 || head->y >= MAX_Y) {
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

	for (int i = 0; i < score + 4; ++i) {
		snake_xs[i] = current->x;
		snake_ys[i] = current->y;
		current = current->next;
	}

	bool valid = false;
	while (!valid) {
		a->x = random_coord(0, MAX_X);
		a->y = random_coord(0, MAX_Y);

		valid = true;
		for (int i = 0; i < score + 4; ++i) {
			if (snake_xs[i] == a->x && snake_ys[i] == a->y) {
				valid = false;
			}
		}
	}
}

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

void get_inputs(mat inputs, snake_part *head, snake_part *tail, apple a)
{
	double head_obstacle_left = head->x + 1;
    	double head_obstacle_right = MAX_X - head->x;
    	double head_obstacle_down = MAX_Y - head->y;
    	double head_obstacle_up = head->y + 1;
    	double head_obstacle_left_down = fmin(head->x, MAX_Y - 1 - head->y) * sqrt(2);
    	double head_obstacle_right_down = fmin(MAX_X - 1 - head->x, MAX_Y - 1 - head->y) * sqrt(2);
    	double head_obstacle_left_up = fmin(head->x, head->y) * sqrt(2);
    	double head_obstacle_right_up = fmin(MAX_X - 1 - head->x, head->y) * sqrt(2);

    	snake_part *current = head->next;
    	while (current) { 
       		if (current->y == head->y) {
           		if (current->x < head->x) {
                		head_obstacle_left = fmin(head_obstacle_left, head->x - current->x);
            		}
           		else if (current->x > head->x) {
                		head_obstacle_right = fmin(head_obstacle_right, current->x - head->x);
            		}
        	}

        	if (current->x == head->x) {
            		if (current->y > head->y) {
                		head_obstacle_down = fmin(head_obstacle_down, current->y - head->y);
           	 	}
           	 	else if (current->y < head->y) {
                		head_obstacle_up = fmin(head_obstacle_up, head->y - current->y);
            		}
        	}

        	if (current->x - head->x == current->y - head->y) {
        		if (current->y < head->y) {
        			head_obstacle_left_down = fmin(head_obstacle_left_down, (head->x - current->x) * sqrt(2));
            		}
            		else {
				head_obstacle_right_up = fmin(head_obstacle_right_up, (current->y - head->y) * sqrt(2));
            		}
        	}

        	if (current->x + head->x == current->y - head->y) {
           		if (current->y < head->y) {
                		head_obstacle_right_down = fmin(head_obstacle_right_down, (current->y - head->y) * sqrt(2));
            		}
            		else {
                		head_obstacle_left_up = fmin(head_obstacle_left_up, (head->x - current->x) * sqrt(2));
          	  	}
        	}

       	 	current = current->next;
	}

	mat_at(inputs, 0, 0) = head_obstacle_left == 1;
	mat_at(inputs, 1, 0) = head_obstacle_right == 1;
	mat_at(inputs, 2, 0) = head_obstacle_down == 1;
	mat_at(inputs, 3, 0) = head_obstacle_up == 1;
	mat_at(inputs, 4, 0) = head_obstacle_left_down == sqrt(2);
	mat_at(inputs, 5, 0) = head_obstacle_right_down == sqrt(2);
	mat_at(inputs, 6, 0) = head_obstacle_left_up == sqrt(2);
	mat_at(inputs, 7, 0) = head_obstacle_right_up == sqrt(2);

	int head_apple_left = 0;
	int head_apple_right = 0;
	int head_apple_down = 0;
	int head_apple_up = 0;
	int head_apple_left_down = 0;
	int head_apple_right_down = 0;
	int head_apple_left_up = 0;
	int head_apple_right_up = 0;

	if (a.y == head->y) {
   		if (a.x < head->x) {
              		head_apple_left = head_obstacle_left > (head->x - a.x);
		}
           	else {
                	head_apple_right = head_obstacle_right > (a.x - head->x);
		}
	}

	if (a.x == head->x) {
    		if (a.y > head->y) {
                	head_apple_down = head_obstacle_down > (a.y - head->y);
		}
         	else {
        		head_apple_up = head_obstacle_up > (head->y - a.y);
		}
        }

        if (a.x - head->x == a.y - head->y) {
		if (a.y < head->y) {
			head_apple_left_down = head_obstacle_left_down > ((head->x - a.x) * sqrt(2));		
		}
		else {
			head_apple_right_up = head_obstacle_right_up > ((a.y - head->y) * sqrt(2));	
		}
        }

        if (a.x + head->x == a.y - head->y) {
		if (a.y < head->y) {
			head_apple_right_down = head_obstacle_right_down > ((a.y - head->y) * sqrt(2));
		}
		else {
			head_apple_left_up = head_obstacle_left_up > ((head->x - a.x) * sqrt(2));
		}
        }

	mat_at(inputs, 8, 0) = head_apple_left;
	mat_at(inputs, 9, 0) = head_apple_right;
	mat_at(inputs, 10, 0) = head_apple_down;
	mat_at(inputs, 11, 0) = head_apple_up;
	mat_at(inputs, 12, 0) = head_apple_left_down;
	mat_at(inputs, 13, 0) = head_apple_right_down;
	mat_at(inputs, 14, 0) = head_apple_left_up;
	mat_at(inputs, 15, 0) = head_apple_right_up;

	mat_at(inputs, 16, 0) = head->dir == LEFT;
	mat_at(inputs, 17, 0) = head->dir == RIGHT;
	mat_at(inputs, 18, 0) = head->dir == DOWN;
	mat_at(inputs, 19, 0) = head->dir == UP;

	mat_at(inputs, 20, 0) = tail->dir == LEFT;
	mat_at(inputs, 21, 0) = tail->dir == RIGHT;
	mat_at(inputs, 22, 0) = tail->dir == DOWN;
	mat_at(inputs, 23, 0) = tail->dir == UP;
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

double fitness(double score, double frames) 
{
	if (score == 0 && frames == 100) {
		return 0;
	}

	return log2(frames + 500 * pow(score, 2) + pow(2, score) - score * frames * .25);
}

int main() 
{
	int score = 0;
	int frames = 0;
	int dead_tracker = 0;
	int dead = 100;
	
	srand(time(0));
	snake_part *head = NULL;
	snake_part *tail = NULL;
	new_snake(&head, &tail);
	apple a;
	new_apple(head, &a, score);
	
	int layers = 4;
	mat topology = mat_alloc(layers, 1);
	mat_at(topology, 0, 0) = 24;
	mat_at(topology, 1, 0) = 20;
	mat_at(topology, 2, 0) = 16;
	mat_at(topology, 3, 0) = 4;
	mat inputs = mat_alloc(mat_at(topology, 0, 0), 1);
	
	int gens = 0;
	int gen_count = 0;
	int epochs = 1500;
	int progress_report = 300;
	int gen_size = 500;
	int best_size = 100;

	specimen *gen = gen_alloc(gen_size, layers, topology);
	for (int i = 0; i < gen_size; ++i) {
		net_rand(gen[i].n, -5, 5);
	}

	specimen *new = gen_alloc(gen_size, layers, topology);
	specimen *best = gen_alloc(best_size, layers, topology);
	specimen *offspring = gen_alloc(best_size, layers, topology);

	free(topology.vals);

	double mutation_rate = 5e-2;
	double mean = 0;
	double stddev = 1;

	bool training = true;
	specimen showcase = gen[0];
	
	while (gens < epochs) {
		while (training) {
			while (!check_collisions(head) && dead_tracker < dead) {
				++frames;
				++dead_tracker;
				get_inputs(inputs, head, tail, a);
				change_direction(inputs, gen[gen_count], head);	
				move_snake(head);

				if (check_eat(head, a)) {
					++score;
					++dead;
					dead_tracker = 0;
					increase_snake(&tail);
					new_apple(head, &a, score);
				}	
			}

			gen[gen_count++].fitness = fitness(score, frames);
			score = 0;
			frames = 0;
			dead = 100;
			dead_tracker = 0;
			destroy_snake(&head);
			head = NULL;
			tail = NULL;
			new_snake(&head, &tail);
			new_apple(head, &a, score);

			if (gen_count == gen_size) {
				gen_count = 0;
				++gens;
				find_best(best, gen, best_size, gen_size);

				double best_fitness = best[0].fitness;
				double selection_avg = 0.0;
				double selection_stddev = 0;
				for (int i = 0; i < best_size; ++i) {
					selection_avg += best[i].fitness;
				}

				selection_avg /= best_size;

				for (int i = 0; i < best_size; ++i) {
					selection_stddev += pow(best[i].fitness - selection_avg, 2);
				}
				
				selection_stddev /= best_size;
				selection_stddev = sqrt(selection_stddev);
				
				printf("Generation: %d | Snakes: %d | Best: %.2f | Selection Avg: %.2f | Selection Stddev: %.2f\n", 
					gens, gen_size, best_fitness, selection_avg, selection_stddev);
	
				for (int i = 0; i < gen_size / best_size; ++i) {
					gen_spx(offspring, best, best_size);

					for (int j = 0; j < best_size; ++j) {
						net_copy(new[i * best_size + j].n, offspring[j].n);
					}

					shuffle(best, sizeof(specimen), best_size);
				}
					
				gen_mutate(new, gen_size, mutation_rate, mean, stddev);
				
				qsort(best, best_size, sizeof(specimen), compare_fitness);
				if (best[0].fitness > showcase.fitness) {
					showcase = best[0];
				}

				if (gens % progress_report == 0) {
					training = false;
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
			printf("Couldn't initialize window %s\n", SDL_GetError());
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
			
			get_inputs(inputs, head, tail, a);
			change_direction(inputs, showcase, head);
			move_snake(head);
		
			if (check_eat(head, a)) {
				++score;
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

		score = 0;
		destroy_snake(&head);
		head = NULL;
		tail = NULL;
		training = true;
		new_snake(&head, &tail);
		new_apple(head, &a, score);
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
	}

	free(inputs.vals);

	gen_destroy(&gen, gen_size);
	gen_destroy(&new, gen_size);
	gen_destroy(&best, best_size);
	gen_destroy(&offspring, best_size);
	
	return 0;
}
