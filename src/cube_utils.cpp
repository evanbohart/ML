#include "cube.h"
#include <cassert>

namespace utils
{
    uint64_t roll_left(uint64_t x, int bits)
    {
        return (x << bits) | (x >> (64 - bits));
    }

    uint64_t roll_right(uint64_t x, int bits)
    {
        return (x >> bits) | (x << (64 - bits));
    }
}
