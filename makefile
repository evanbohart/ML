CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -g3 -O3 -march=native -fopenmp -ffast-math \
		 -I./include -I./SDL/x86_64-w64-mingw32/include/SDL2/
LDFLAGS = -L./SDL/x86_64-w64-mingw32/lib/ -lSDL2

NN_SRCS = src/nn/nn.c src/nn/block/res_block.c src/nn/layer/dense_layer.c \
		  src/nn/layer/conv_layer.c src/nn/layer/maxpool_layer.c src/nn/layer/flatten_layer.c \
		  src/nn/layer/dropout_layer.c src/nn/layer/batchnorm_layer.c src/nn/layer/sig_layer.c \
		  src/nn/layer/tanh_layer.c src/nn/layer/relu_layer.c src/nn/layer/gelu_layer.c \
		  src/nn/layer/softmax_layer.c src/nn/tens/mat.c src/nn/tens/tens3D.c src/nn/tens/tens4D.c \
		  src/nn/utils/utils.c src/nn/utils/funcs.c
NN_OBJS = $(NN_SRCS:src/nn/%.c=obj/nn/%.o)

IMG_SRCS = src/img.c
IMG_OBJS = $(IMG_SRCS:src/%.c=obj/%.o)

all: img

obj:
	mkdir obj/nn/tens
	mkdir obj/nn/layer
	mkdir obj/nn/block
	mkdir obj/nn/utils

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

test: $(NN_OBJS) obj/test.o
	$(CC) $(CFLAGS) $(NN_OBJS) obj/test.o -o test -lm

img: $(IMG_OBJS) $(NN_OBJS)
	$(CC) $(CFLAGS) $(IMG_OBJS) $(NN_OBJS) -o img -lm

clean:
	rm -rf obj test
