CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -g3 -O3 -march=native -fopenmp -ffast-math -I./include -I./SDL/x86_64-w64-mingw32/include/SDL2/
LDFLAGS = -L./SDL/x86_64-w64-mingw32/lib/ -lSDL2

IMG_SRCS = src/img.c src/mat.c src/tens3D.c src/tens4D.c src/dense_layer.c src/conv_layer.c src/maxpool_layer.c \
		   src/flatten_layer.c src/dense_dropout_layer.c src/conv_dropout_layer.c src/nn.c src/utils.c
IMG_OBJS = $(IMG_SRCS:src/%.c=obj/%.o)

CUBE_C_SRCS = src/mat.c src/tens.c src/net.c src/cnet.c src/utils.c
CUBE_CPP_SRCS = src/cube.cpp src/cube_model.cpp src/cube_ai.cpp src/cube_utils.cpp
CUBE_OBJS = $(CUBE_C_SRCS:src/%.c=obj/%.o) $(CUBE_CPP_SRCS:src/%.cpp=obj/%.o)

CHESS_SRCS = src/mat.c src/tens.c src/dense_layer.c src/conv_layer.c src/nn.c src/utils.c \
			 src/board.c src/move.c src/attack.c src/magic.c src/pin.c src/test.c
CHESS_OBJS = $(CHESS_SRCS:src/%.c=obj/%.o)

all: img

obj:
	mkdir -p obj

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/%.o: src/%.cpp | obj
	$(CXX) $(CFLAGS) -c $< -o $@

img: $(IMG_OBJS)
	$(CC) $(CFLAGS) $(IMG_OBJS) -o img -lm

cube: $(CUBE_OBJS)
	$(CXX) $(CFLAGS) $(CUBE_OBJS) -o cube $(LDFLAGS)

chess: $(CHESS_OBJS)
	$(CC) $(CFLAGS) $(CHESS_OBJS) -o chess

clean:
	rm -rf obj snake cube chess
