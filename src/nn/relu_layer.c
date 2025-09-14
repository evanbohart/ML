#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer relu_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b)
{
    relu_layer *rl = malloc(sizeof(relu_layer));

    rl->x_r = x_r;
    rl->x_c = x_c;
    rl->x_d = x_d;
    rl->x_b = x_b;

    rl->x_cache = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = rl;

    l.forward = relu_forward;
    l.backprop = relu_backprop;
    l.destroy = relu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void relu_forward(layer l, tens x, tens *y)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(x.dims[R] == rl->x_r);
    assert(x.dims[C] == rl->x_c);
    assert(x.dims[D] == rl->x_d);
    assert(x.dims[B] == rl->x_b);

    *y = tens_alloc(rl->x_r, rl->x_c, rl->x_d, rl->x_b);

    tens_copy(rl->x_cache, x);

    tens_func(*y, x, relu);
}

void relu_backprop(layer l, tens dy, tens *dx, float rate)
{
    relu_layer *rl = (relu_layer *)l.data;

    assert(dy.dims[R] == rl->x_r);
    assert(dy.dims[C] == rl->x_c);
    assert(dy.dims[D] == rl->x_d);
    assert(dy.dims[B] == rl->x_b);

    *dx = tens_alloc(rl->x_r, rl->x_c, rl->x_d, rl->x_b);

    tens_func(*dx, rl->x_cache, drelu);
    tens_had(*dx, *dx, dy);
}

void relu_destroy(layer l)
{
    relu_layer *rl = (relu_layer *)l.data;

    tens_destroy(rl->x_cache);

    free(rl);
}
