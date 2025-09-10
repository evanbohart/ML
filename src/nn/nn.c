#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "nn.h"

nn nn_alloc(int max_layers)
{
    nn n;

    n.max_layers = max_layers;
    n.num_layers = 0;

    n.layers = malloc(max_layers * sizeof(layer));

    return n;
}

void nn_add_layer(nn *n, layer l)
{
    assert(n->num_layers != n->max_layers);

    n->layers[n->num_layers++] = l;
}

void nn_forward(nn n, tens x, tens *y)
{
    tens x_current = x;
    tens y_current;

    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].forward(n.layers[i], x_current, &y_current);

        if (i > 0) {
            tens_destroy(x_current);
        }

        x_current = y_current;
    }

    *y = y_current;
}

void nn_backprop(nn n, tens dy, tens *dx, float rate)
{
    tens dy_current = dy;
    tens dx_current;

    for (int i = n.num_layers - 1; i >= 0; --i) {
        n.layers[i].backprop(n.layers[i], dy_current, &dx_current, rate);

        if (i < n.num_layers - 1) {
            tens_destroy(dy_current);
        }

        dy_current = dx_current;
    }

    *dx = dx_current;
}

void nn_destroy(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].destroy(n.layers[i]);
    }
}

void nn_init(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].init(n.layers[i]);
    }
}

void nn_print(nn n)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].print(n.layers[i]);
    }
}

void nn_save(nn n, FILE *f)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].save(n.layers[i], f);
    }
}

void nn_load(nn n, FILE *f)
{
    for (int i = 0; i < n.num_layers; ++i) {
        n.layers[i].load(n.layers[i], f);
    }
}
