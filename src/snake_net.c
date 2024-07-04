#include "snake.h"
#include <math.h>

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

	mat_at(inputs, 8, 0) = head_obstacle_left < 5;
	mat_at(inputs, 9, 0) = head_obstacle_right < 5;
	mat_at(inputs, 10, 0) = head_obstacle_down < 5;
	mat_at(inputs, 11, 0) = head_obstacle_up < 5;
	mat_at(inputs, 12, 0) = head_obstacle_left_down < (5 * sqrt(2));
	mat_at(inputs, 13, 0) = head_obstacle_right_down < (5 * sqrt(2));
	mat_at(inputs, 14, 0) = head_obstacle_left_up < (5 * sqrt(2));
	mat_at(inputs, 15, 0) = head_obstacle_right_up < (5 * sqrt(2));


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

	mat_at(inputs, 16, 0) = head_apple_left;
	mat_at(inputs, 17, 0) = head_apple_right;
	mat_at(inputs, 18, 0) = head_apple_down;
	mat_at(inputs, 19, 0) = head_apple_up;
	mat_at(inputs, 20, 0) = head_apple_left_down;
	mat_at(inputs, 21, 0) = head_apple_right_down;
	mat_at(inputs, 22, 0) = head_apple_left_up;
	mat_at(inputs, 23, 0) = head_apple_right_up;

	mat_at(inputs, 24, 0) = head->dir == LEFT;
	mat_at(inputs, 25, 0) = head->dir == RIGHT;
	mat_at(inputs, 26, 0) = head->dir == DOWN;
	mat_at(inputs, 27, 0) = head->dir == UP;

	mat_at(inputs, 28, 0) = tail->dir == LEFT;
	mat_at(inputs, 29, 0) = tail->dir == RIGHT;
	mat_at(inputs, 30, 0) = tail->dir == DOWN;
	mat_at(inputs, 31, 0) = tail->dir == UP;
}

void change_direction(mat inputs, net n, snake_part *head) 
{
	feed_forward(n, inputs, relu);

	int output = 0;
	double max = 0;
	for (int i = 0; i < 4; ++i) {
		if (mat_at(n.acts[n.layers - 2], i, 0) > max) {
			output = i;
			max = mat_at(n.acts[n.layers - 2], i, 0);
		}
	}

	head->dir = output;
}

double fitness(double score, double frames) 
{
	if (score == 0 && frames == 100) {
		return 0;
	}

	return sqrt(frames + 500 * pow(score, 2) - (score > 0 ? frames / score : 0));
}
