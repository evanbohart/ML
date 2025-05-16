CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -g3 -O1 -I./include -I./SDL/x86_64-w64-mingw32/include/SDL2/
LDFLAGS = -L./SDL/x86_64-w64-mingw32/lib/ -lSDL2

IMG_SRCS = src/img.c src/mat.c src/tens.c src/net.c src/cnet.c src/utils.c
IMG_OBJS = $(IMG_SRCS:src/%.c=obj/%.o)

CUBE_C_SRCS = src/mat.c src/tens.c src/net.c src/cnet.c src/utils.c
CUBE_CPP_SRCS = src/cube.cpp src/cube_model.cpp src/cube_ai.cpp src/cube_utils.cpp
CUBE_OBJS = $(CUBE_C_SRCS:src/%.c=obj/%.o) $(CUBE_CPP_SRCS:src/%.cpp=obj/%.o)

CHESS_SRCS = src/chess.c src/magic.c src/test.c
CHESS_OBJS = $(CHESS_SRCS:src/%.c=obj/%.o)

all: cube img chess

obj:
	mkdir -p obj

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/%.o: src/%.cpp | obj
	$(CXX) $(CFLAGS) -c $< -o $@

img: $(IMG_OBJS)
	$(CC) $(IMG_OBJS) -o img

cube: $(CUBE_OBJS)
	$(CXX) $(CUBE_OBJS) -o cube $(LDFLAGS)

chess: $(CHESS_OBJS)
	$(CC) $(CHESS_OBJS) -o chess

clean:
	rm -rf obj snake cube
