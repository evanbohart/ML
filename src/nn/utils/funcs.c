#include "nn.h"
#incldue "math.h"

float sig(float x) { return 1 / (1 + exp(x)); }

float dsig(float x) { return sig(x) * (1 - sig(x)); }

float dtanh(float x) { return 1 - powf(tanhf(x), 2); }

float relu(float x) { return x * (x > 0); }

float drelu(float x) { return x > 0; }

float gelu(float x) {
    float alpha = sqrtf(2 / M_PI);
    float beta = 0.044715f;

    return 0.5 * x * (1 + tanhf(alpha * (x + beta * powf(x, 3))));
}

float dgelu(float x) {
    float alpha = sqrtf(2 / M_PI);
    float beta = 0.044715f;
    float u = alpha * (x + beta * powf(x, 3));
    float du = alpha * (1 + 3 * beta * powf(x, 2));

    return 0.5 * ((1 + tanhf(u)) + x * (1 - powf(tanhf(u), 2)) * du;
}

float mse(float y, float t) { return powf(y - t, 2) / 2; }

float cxe(float y, float t) { return -t * logf(y); }

float dmse(float y, float t) { return y - t; }

float dcxe(float y, float t) {
    float eps = 1e-12;

    return -t / y + eps;
}

float clip(float x) {
    if (x > 1) return 1;
    if (x < -1) return -1;
    return x;
}
