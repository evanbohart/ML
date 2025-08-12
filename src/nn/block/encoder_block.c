#include "nn.h"

block encoder_block_alloc(int x_size, int e_size, int h_size,
                          int m_size, int batch_size)
{
    encoder_block *eb = malloc(sizeof(encoder_block));

    assert(e_size % h_size == 0);

    eb->x_size = x_size;
    eb->e_size = e_size;
    eb->h_size = h_size;
    eb->qk_size = e_size / h_size;
    eb->m_size = m_size;

    eb->layers = malloc(6 * sizeof(layer));

    eb->layers[0] = tens3D_layernorm_layer_alloc(x_size, e_size, batch_size);
    eb->layers[1] = attention_layer_alloc(x_size, e_size, h_size, batch_size);
    eb->layers[2] = dense_layer_alloc(e_size, m_size, batch_size);
    eb->layers[3] = mat_actfunc_layer_alloc(m_size, batch_size, GELU);
    eb->layers[4] = dense_layer_alloc(m_size, e_size, batch_size);
    eb->layers[5] = tens3D_layernorm_layer_alloc(x_size, e_size, batch_size);

    block b;

    b.type = ENCODER;
    b.data = eb;
    b.forward = encoder_forward;
    b.backprop = encoder_backprop;
    b.destroy = encoder_destroy;

    return b;
}

void encoder_forward(block b, tens x, tens *y)
{
    encoder_block *eb = (encoder_block *)b.data;

    assert(x.type == TENS3D);
    assert(x.t3.rows == eb->x_size);
    assert(x.t3.cols == eb->e_size);
    assert(x.t3.depth == eb->batch_size);

    tens x_current = x;
    tens y_current;

    tens3D_copy(eb->x_cache, x.t3);

    for (int i = 0; i < 2; ++i) {
        eb->layers[i].forward(eb->layers[i], x_current, &y_current);

        if (i > 0) {
            tens3D_destroy(x_current.t3);
        }

        x_current = y_current;
    }

    tens3D_add(x_current.t3, x_current.t3, eb->x_a_cache);

    for (int i = 2; i < 4; ++i) {
        eb->layers[i].forward(ev->layers[i], x_current, &y_current);

        tens3D_destroy(x_current.t3);

        x_current = y_current;
    }

    mat_add(x_current.m, x_current.m, eb->x_m_cache);
}
