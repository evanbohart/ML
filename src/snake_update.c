#include "snake.h"
#include <stdlib.h>
#include <assert.h>

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
    while (current) {
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
