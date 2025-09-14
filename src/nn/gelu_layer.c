#include <stdlib.h>
#include <assert.h>
#include "nn.h"

layer gelu_layer_alloc(int x_r, int x_c,
                       int x_d, int x_b)
{
    gelu_layer *gl = malloc(sizeof(gelu_layer));

    gl->x_r = x_r;
    gl->x_c = x_c;
    gl->x_d = x_d;
    gl->x_b = x_b;

    gl->x_cache = tens_alloc(x_r, x_c, x_d, x_b);

    layer l;

    l.data = gl;

    l.forward = gelu_forward;
    l.backprop = gelu_backprop;
    l.destroy = gelu_destroy;

    l.init = NULL;
    l.print = NULL;
    l.save = NULL;
    l.load = NULL;

    return l;
}

void gelu_forward(layer l, tens x, tens *y)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(x.dims[R] == gl->x_r);
    assert(x.dims[C] == gl->x_c);
    assert(x.dims[D] == gl->x_d);
    assert(x.dims[B] == gl->x_b);

    *y = tens_alloc(gl->x_r, gl->x_c, gl->x_d, gl->x_b);

    tens_copy(gl->x_cache, x);

    tens_func(*y, x, gelu);
}

void gelu_backprop(layer l, tens dy, tens *dx, float rate)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    assert(dy.dims[R] == gl->x_r);
    assert(dy.dims[C] == gl->x_c);
    assert(dy.dims[D] == gl->x_d);
    assert(dy.dims[B] == gl->x_b);

    *dx = tens_alloc(gl->x_r, gl->x_c, gl->x_d, gl->x_b);

    tens_func(*dx, gl->x_cache, dgelu);
    tens_had(*dx, *dx, dy);
}

void gelu_destroy(layer l)
{
    gelu_layer *gl = (gelu_layer *)l.data;

    tens_destroy(gl->x_cache);

    free(gl);
}
