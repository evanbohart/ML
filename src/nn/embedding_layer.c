#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "nn.h"

layer embedding_layer_alloc(int x_size, int batch_size, int e_size, int v_size)
{
    embedding_layer *el = malloc(sizeof(embedding_layer));

    el->x_size = x_size;
    el->batch_size = batch_size;
    el->e_size = esize;
    el->v_size = v_size;

    el->e = mat_alloc(v_size, e_size);
    el->p = mat_alloc(x_size, e_size);

    layer l;

    l.type = EMBEDDING;
    l.data = el;
    l.forward = embedding_forward;
    l.backprop = embedding_backprop;
    l.destroy = embedding_destroy;
    l.init = embedding_init;
    l.save = embedding_save;
    l.load = embedding_load;

    return l;
}

void embedding_forward(layer l, void *x, void **y)
{
    embedding_layer *el = (embedding_layer *)l.data;

    mat *mat_x = (mat *)x;

    assert(mat_x->rows == el->x_size);
    assert(mat_x->cols == el->batch_size);

    tens3D *tens3D_y = malloc(sizeof(tens3D));
    *tens3D_y = tens3D_alloc(el->x_size, el->e_size, el->batch_size);

    for (int i = 0; i < el->x_size; ++i) {
        for (int j = 0; j < el->e_size; ++j) {
            for (int k = 0; k < el->batch_size; ++k) {
                int index = mat_at(*mat_x, i, k);

                tens3D_at(*tens3D_y, i, j, k) = mat_at(el->e, index, j) + mat_at(el->p, i, j);
            }
        }
    }

    *y = tens3D_y;
}

void embedding_destroy(layer l)
{
    embedding_layer *el = (embedding_layer *)l.data;

    free(el->e.vals);
    free(el->p.vals);

    free(el);
}

void embedding_init(layer l)
{
    embedding_layer *el = (embedding_layer *)l.data;

    float range = 1.0f / sqrtf(el->y_size);

    for (int i = 0; i < el->v_size; ++i) {
        for (int j = 0; j < el->y_size; ++j) {
            mat_at(el->e, i, j) = rand_float(-range, range);
        }
    }

    for (int i = 0; i < el->x_size; ++i) {
        for (int j = 0; j < el->y_size; ++j) {
            float denom = pow(10 * 1000.0f, (float)(j / 2) / el->y_size);

            if (j % 2 == 0) {
                mat_at(el->p, i, j) = sinf(i / denom);
            }
            else {
                mat_at(el->p, i, j) = cosf(i / denom);
            }
        }
    }
}

void embedding_save(layer l, FILE *f)
{
    embedding_layer *el = (embedding_layer *)l.data;

    mat_save(el->e, f);
    mat_save(el->p, f);
}

void embedding_load(layer l, FILE *f)
{
    embedding_layer *el = (embedding_layer *)l.data;

    mat_load(el->e, f);
    mat_load(el->p, f);
}
