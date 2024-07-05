#include "nn.h"
#include "utils.h"
#include "snake.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

int train(int epochs, char *file_name)
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
    mat_at(topology, 0, 0) = 32;
    mat_at(topology, 1, 0) = 28;
    mat_at(topology, 2, 0) = 20;
    mat_at(topology, 3, 0) = 4;
    mat inputs = mat_alloc(mat_at(topology, 0, 0), 1);
	
    int gens = 0;
    int gen_count = 0;
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

    specimen showcase = gen[0];
	
    while (gens < epochs) {
        while (!check_collisions(head) && dead_tracker < dead) {
            ++frames;
            ++dead_tracker;
            get_inputs(inputs, head, tail, a);
            change_direction(inputs, gen[gen_count].n, head);	
            move_snake(head);

            if (check_eat(head, a)) {
                ++score;
                ++dead;
                dead_tracker = 0;
                increase_snake(&tail);

                if (score < 396) {
                    new_apple(head, &a, score);
                }
                else {
                    dead = 0;
                }
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

                gen_copy(&gen, new, gen_size);
            }
        }
    }

    char path[FILENAME_MAX];
    get_path(path, file_name);
    clear_bin(path);
    FILE *f = fopen(path, "ab");
    net_save(showcase.n, f);

    free(inputs.vals);
    destroy_snake(&head);

    gen_destroy(&gen, gen_size);
    gen_destroy(&new, gen_size);
    gen_destroy(&best, best_size);
    gen_destroy(&offspring, best_size);

    return 0;
}

int showcase(char *file_name)
{
    int score = 0;
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
    mat_at(topology, 0, 0) = 32;
    mat_at(topology, 1, 0) = 28;
    mat_at(topology, 2, 0) = 20;
    mat_at(topology, 3, 0) = 4;
    mat inputs = mat_alloc(mat_at(topology, 0, 0), 1);
	
    net n = net_alloc(layers, topology);

    char path[FILENAME_MAX];
    get_path(path, file_name);

    FILE *f = fopen(path, "rb");
    net_load(&n, &f);
    fclose(f);

    free(topology.vals);

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
		
    while (!check_collisions(head) && dead_tracker < dead && !quit) {
        ++dead_tracker;

        SDL_SetRenderDrawColor(renderer, 0x22, 0x22, 0x22, 0xFF);
        SDL_RenderClear(renderer);

        get_inputs(inputs, head, tail, a);
        change_direction(inputs, n, head);
        move_snake(head);

        if (check_eat(head, a)) {
            ++score;
            ++dead;
            increase_snake(&tail);
			
            if (score < 396) {
                dead_tracker = 0;
                new_apple(head, &a, score);
            }
            else {
                dead = 0;
            }
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

    destroy_snake(&head);
    net_destroy(&n);
    free(inputs.vals);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

int main(int argc, char **argv) 
{
    if (argc < 2) {
        printf("Usage: %s <mode> <options>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0 && argc < 4) {
        printf("Usage: %s %s <epochs> <file_name>\n", argv[0], argv[1]);
        return 1;
    }
    else if (strcmp(argv[1], "train") == 0) {
        return train(atoi(argv[2]), argv[3]);
    }

    if (strcmp(argv[1], "showcase") == 0 && argc < 3) {
        printf("Usage: %s %s <file_name>\n", argv[0], argv[1]);
        return 1;
    }
    else if (strcmp(argv[1], "showcase") == 0) {
        return showcase(argv[2]);
    }

    printf("Usage %s <mode> <options>\n", argv[0]);
    return 1;
}
