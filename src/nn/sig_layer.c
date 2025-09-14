#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer sig_layer_alloc(int x_r, int x_c,
                      int x_d, int x_b)
{
    sig_layer *sl = malloc(sizeof(sig_layer));

    sl->x_r = x_r;
    sl->x_c = x_c;
    sl->x_d = x_d;
    sl->x_b = x_b;

    sl->x_cache = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = sl;

    l.forward = sig_forward;
    l.backprop = sig_backprop;
    l.destroy = sig_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void sig_forward(layer l, tens x, tens *y)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(x.dims[R] == sl->x_r);
    assert(x.dims[C] == sl->x_c);
    assert(x.dims[D] == sl->x_d);
    assert(x.dims[B] == sl->x_b);

    *y = tens_alloc(sl->x_r, sl->x_c, sl->x_d, sl->x_b);

    tens_copy(sl->x_cache, x);

    tens_func(*y, x, sig);
}

void sig_backprop(layer l, tens dy, tens *dx, float rate)
{
    sig_layer *sl = (sig_layer *)l.data;

    assert(dy.dims[R] == sl->x_r);
    assert(dy.dims[C] == sl->x_c);
    assert(dy.dims[D] == sl->x_d);
    assert(dy.dims[B] == sl->x_b);

    *dx = tens_alloc(sl->x_r, sl->x_c, sl->x_d, sl->x_b);

    tens_func(*dx, sl->x_cache, dsig);
    tens_had(*dx, *dx, dy);
}

void sig_destroy(layer l)
{
    sig_layer *sl = (sig_layer *)l.data;

    tens_destroy(sl->x_cache);

    free(sl);
}
