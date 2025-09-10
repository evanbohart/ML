#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer reshape_layer_alloc(int x_rows, int x_cols, int x_depth,
                          int x_batches, int y_rows, int y_cols,
                          int y_depth, int y_batches)
{
    reshape_layer *rl = malloc(sizeof(reshape_layer));

    int x_elements = x_rows * x_cols * x_depth * x_batches;
    int y_elements = y_rows * y_cols * y_depth * y_batches;

    assert(x_elements == y_elements);

    rl->x_rows = x_rows;
    rl->x_cols = x_cols;
    rl->x_depth = x_depth;
    rl->x_batches = x_batches;

    rl->y_rows = y_rows;
    rl->y_cols = y_cols;
    rl->y_depth = y_depth;
    rl->y_batches = y_batches;

    layer l;

    l.data = rl;

    l.forward = reshape_forward;
    l.backprop = reshape_backprop;
    l.destroy = reshape_destroy;

    return l;
}

void reshape_forward(layer l, tens x, tens *y)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    assert(x.rows == rl->x_rows);
    assert(x.cols == rl->x_cols);
    assert(x.depth == rl->x_depth);
    assert(x.batches == rl->x_batches);

    *y = tens4D_alloc(rl->y_rows, rl->y_cols, rl->y_depth, rl->y_batches);

    tens_reshape(*y, x);
}

void reshape_backprop(layer l, tens dy, tens *dx, float rate)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    assert(dy.rows == rl->y_rows);
    assert(dy.cols == rl->y_cols);
    assert(dy.depth == rl->y_depth);
    assert(dy.batches == rl->y_batches);

    *dx = tens4D_alloc(rl->x_rows, rl->x_cols, rl->x_depth, rl->x_batches);

    tens_reshape(*dx, dy);
}

void reshape_destroy(layer l)
{
    reshape_layer *rl = (reshape_layer *)l.data;

    free(rl);
}
