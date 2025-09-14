#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer reshape_layer_alloc(int x_r, int x_c, int x_d,
                          int x_b, int y_r, int y_c,
                          int y_d, int y_b)
{
    reshape_layer *rl = malloc(sizeof(reshape_layer));

    int x_elements = x_r * x_c * x_d * x_b;
    int y_elements = y_r * y_c * y_d * y_b;

    assert(x_elements == y_elements);

    rl->x_r = x_r;
    rl->x_c = x_c;
    rl->x_d = x_d;
    rl->x_b = x_b;

    rl->y_r = y_r;
    rl->y_c = y_c;
    rl->y_d = y_d;
    rl->y_b = y_b;

    layer l;

    l.data = rl;

    l.forward = reshape_forward;
    l.backprop = reshape_backprop;
    l.destroy = reshape_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void reshape_forward(layer l, tens x, tens *y)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    assert(x.dims[R] == rl->x_r);
    assert(x.dims[C] == rl->x_c);
    assert(x.dims[D] == rl->x_d);
    assert(x.dims[B] == rl->x_b);

    *y = tens_alloc(rl->y_r, rl->y_c, rl->y_d, rl->y_b);

    tens_reshape(*y, x);
}

void reshape_backprop(layer l, tens dy, tens *dx, float rate)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    assert(dy.dims[R] == rl->y_r);
    assert(dy.dims[C] == rl->y_c);
    assert(dy.dims[D] == rl->y_d);
    assert(dy.dims[B] == rl->y_b);

    *dx = tens_alloc(rl->x_r, rl->x_c, rl->x_d, rl->x_b);

    tens_reshape(*dx, dy);
}

void reshape_destroy(layer l)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    free(rl);
}
