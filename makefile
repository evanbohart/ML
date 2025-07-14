CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -g3 -O3 -march=native -fopenmp -ffast-math -I./include -I./SDL/x86_64-w64-mingw32/include/SDL2/
LDFLAGS = -L./SDL/x86_64-w64-mingw32/lib/ -lSDL2

NN_SRCS = src/mat.c src/tens3D.c src/tens4D.c src/nn.c src/dense_layer.c src/conv_layer.c \
		  src/dense_dropout_layer.c src/conv_dropout_layer.c src/recurrent_layer.c src/lstm_layer.c \
		  src/maxpool_layer.c src/concat_layer.c src/flatten_layer.c src/utils.c
NN_OBJS = $(NN_SRCS:src/%.c=obj/%.o)

IMG_SRCS = src/img.c
IMG_OBJS = $(IMG_SRCS:src/%.c=obj/%.o)

all: test img

obj:
	mkdir obj

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/%.o: src/%.cpp | obj
	$(CC) $(CFLAGS) -c $< -o $@

test: $(NN_OBJS) obj/test.o
	$(CC) $(CFLAGS) $(NN_OBJS) obj/test.o -o test -lm

img: $(IMG_OBJS) $(NN_OBJS)
	$(CC) $(CFLAGS) $(IMG_OBJS) $(NN_OBJS) -o img -lm

clean:
	rm -rf obj test
