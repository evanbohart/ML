#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

nn nn_init(void)
{
    nn n;

    n.head = NULL;
    n.tail = NULL;

    return n;
}

void nn_add_layer(nn *n, layer l)
{
    if (!n.head) {
        n->head = malloc(sizeof(node));
        n->head->type = LAYER;
        n->head->l = l;

        n->head->next = NULL;
        n->head->prev = NULL;
        n->tail = n->head;
    }
    else {
        n->tail->next = malloc(sizeof(node));
        n->tail->next->type = LAYER;
        n->tail->next->l = l;

        node *temp = n->tail;
        n->tail = n->tail->next;
        n->tail->next = NULL;
        n->tail->prev = temp;
    }
}

void nn_add_block(nn *n, block b)
{
    if (!n.head) {
        n->head = malloc(sizeof(node));
        n->head->type = BLOCK;
        n->head->b = b;

        n->head->next = NULL;
        n->head->prev = NULL;
        n->tail = n->head;
    }
    else {
        n->tail->next = malloc(sizeof(node));
        n->tail->next->type = BLOCK;
        n->tail->next->b = b;

        node *temp = n->tail;
        n->tail = n->tail->next;
        n->tail->next = NULL;
        n->tail->prev = temp;
    }
}

void nn_forward(nn n, tens x, tens *y)
{
    tens x_current = x;
    tens y_current;

    node *current = n.head;
    bool first = true;

    while (current) {
        switch (current->type) {
            case LAYER:
                current->l.forward(current->l, x_current, &y_current);
                break;
            case BLOCK:
                current->b.forward(current->b, x_current, &y_current);
        }

        if (!first) {
            switch (x_current.type) {
                case MAT:
                    free(x_current.m.vals);
                    break;
                case TENS3D:
                    tens3D_destroy(x_current.t3);
                    break;
                case TENS4D:
                    tens4D_destroy(x_current.t4);
            }
        }

        first = false;

        x_current = y_current;

        current = current->next;
    }
}

void nn_backprop(nn n, tens dy, tens *dx, float rate)
{
    tens dy_current = dy;
    tens dx_current;

    node *current = n.tail;
    bool first = true;

    while (current) {
        switch (current->type) {
            case LAYER:
                current->l.backprop(current->l, dy_current, &dx_current, rate);
                break;
            case BLOCK:
                current->b.backprop(current->b, dy_current, &dx_current, rate);
        }

        if (!first) {
            switch (dy_current.type) {
                case MAT:
                    free(dy_current.m.vals);
                    break;
                case TENS3D:
                    tens3D_destroy(dy_current.t3);
                    break;
                case TENS4D:
                    tens4D_destroy(dy_current.t4);
            }
        }

        first = false;

        dy_current = dx_current;

        current = current->prev;
    }
}

void nn_destroy(nn n)
{
    node *current = n.head;

    while (current)
    {
        switch (current->type) {
            case LAYER:
                current->l.destroy(current->l);
                break;
            case BLOCK:
                current->b.destroy(current->b);
                break;
        }

        node *temp = current;
        current = current->next;

        free(temp);
    }
}

void nn_init(nn n)
{
    
}

void nn_print(nn n)
{
    
}

void nn_save(nn n, FILE *f)
{
    
}

void nn_load(nn n, FILE *f)
{
}
