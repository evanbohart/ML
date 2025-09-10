CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -g3 -O3 -march=native -fopenmp -ffast-math \
		 -I./include -I./SDL/x86_64-w64-mingw32/include/SDL2/
LDFLAGS = -L./SDL/x86_64-w64-mingw32/lib/ -lSDL2

NN_SRCS = src/nn/nn.c src/nn/res_block.c src/nn/dense_layer.c \
		  src/nn/conv_layer.c src/nn/maxpool_layer.c src/nn/reshape_layer.c \
		  src/nn/dropout_layer.c src/nn/batchnorm_layer.c src/nn/sig_layer.c \
		  src/nn/tanh_layer.c src/nn/relu_layer.c src/nn/gelu_layer.c \
		  src/nn/softmax_layer.c src/nn/tens.c src/nn/utils.c src/nn/funcs.c
NN_OBJS = $(NN_SRCS:src/nn/%.c=obj/nn/%.o)

IMG_SRCS = src/img.c
IMG_OBJS = $(IMG_SRCS:src/%.c=obj/%.o)

all: img

obj:
	mkdir -p obj/nn/tens \
	mkdir -p obj/nn/layer \
	mkdir -p obj/nn/block \
	mkdir -p obj/nn/utils

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

test: $(NN_OBJS) obj/test.o
	$(CC) $(CFLAGS) $(NN_OBJS) obj/test.o -o test -lm

img: $(IMG_OBJS) $(NN_OBJS)
	$(CC) $(CFLAGS) $(IMG_OBJS) $(NN_OBJS) -o img -lm

clean:
	rm -rf obj test
