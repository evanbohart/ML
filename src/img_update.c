#include "img.h"

void set_pixel(bool *pixels, int x, int y)
{
    pixels[x + MAX_X * y] = true;
}
