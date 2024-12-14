#ifndef SNAKE_H
#define SNAKE_H

#include "nn.h"
#include <stdbool.h>
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

typedef struct snake_part 
{
  int x;
  int y;
  int dir;
  struct snake_part *next;
} snake_part;

typedef struct apple
{
  int x;
  int y;
} apple;

int random_coord(int min, int max);
void increase_snake(snake_part **tail);
void new_snake(snake_part **head, snake_part **tail);
void destroy_snake(snake_part **head);
void move_snake(snake_part *head);
bool check_collisions(snake_part *head);
bool check_eat(snake_part *head, apple a);
void new_apple(snake_part *head, apple *a, int score);

void render_background(SDL_Renderer *renderer);
void render_snake(SDL_Renderer *renderer, snake_part *head);
void render_apple(SDL_Renderer *renderer, apple a);

void get_inputs(mat inputs, snake_part *head, snake_part *tail, apple a);
void change_direction(mat inputs, net n, snake_part *head);
double fitness(double score, double frames);

typedef struct specimen {
	double fitness;
	net n;
} specimen;

specimen *gen_alloc(int size, int layers, mat topology);
void gen_destroy(specimen **gen, int size);
void gen_copy(specimen **destination, specimen *gen, int size);
int compare_fitness(const void *p, const void *d);
void find_best(specimen *desintation, specimen *gen, int new_size, int current_size);
void gen_spx(specimen *destination, specimen *gen, int size);
void gen_mutate(specimen *gen, int size, double rate, double mean, double std_dev);

#endif
