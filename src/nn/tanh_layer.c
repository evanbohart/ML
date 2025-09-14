#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn.h"

layer tanh_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b)
{
    tanh_layer *tl = malloc(sizeof(tanh_layer));

    tl->x_r = x_r;
    tl->x_c = x_c;
    tl->x_d = x_d;
    tl->x_b = x_b;

    tl->x_cache = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = tl;

    l.forward = tanh_forward;
    l.backprop = tanh_backprop;
    l.destroy = tanh_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void tanh_forward(layer l, tens x, tens *y)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(x.dims[R] == tl->x_r);
    assert(x.dims[C] == tl->x_c);
    assert(x.dims[D] == tl->x_d);
    assert(x.dims[B] == tl->x_b);

    *y = tens_alloc(tl->x_r, tl->x_c, tl->x_d, tl->x_b);

    tens_copy(tl->x_cache, x);

    tens_func(*y, x, tanhf);
}

void tanh_backprop(layer l, tens dy, tens *dx, float rate)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    assert(dy.dims[R] == tl->x_r);
    assert(dy.dims[C] == tl->x_c);
    assert(dy.dims[D] == tl->x_d);
    assert(dy.dims[B] == tl->x_b);

    *dx = tens_alloc(tl->x_r, tl->x_c, tl->x_d, tl->x_b);

    tens_func(*dx, tl->x_cache, dtanh);
    tens_had(*dx, *dx, dy);
}

void tanh_destroy(layer l)
{
    tanh_layer *tl = (tanh_layer *)l.data;

    tens_destroy(tl->x_cache);

    free(tl);
}
