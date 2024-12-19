CC = gcc
CXX = g++

CFLAGS = -Wall -Wextra -I./include -I./../SDL/include
LDFLAGS = -L./../SDL/lib/x64 -lSDL2

SNAKE_SRCS = src/snake.c src/mat.c src/net.c src/gen.c src/snake_update.c src/snake_render.c src/snake_net.c src/utils.c
SNAKE_OBJS = $(SNAKE_SRCS:src/%.c=obj/%.o)

CUBE_C_SRCS = src/mat.c src/net.c src/utils.c
CUBE_CPP_SRCS = src/cube.cpp src/cube_model.cpp src/cube_ai.cpp src/cube_utils.cpp
CUBE_OBJS = $(CUBE_C_SRCS:src/%.c=obj/%.o) $(CUBE_CPP_SRCS:src/%.cpp=obj/%.o)

all: cube

obj:
	mkdir -p obj

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/%.o: src/%.cpp | obj
	$(CXX) $(CFLAGS) -c $< -o $@

snake: $(SNAKE_OBJS)
	$(CC) $(SNAKE_OBJS) -o snake $(LDFLAGS)

cube: $(CUBE_OBJS)
	$(CXX) $(CUBE_OBJS) -o cube $(LDFLAGS)

clean:
	rm -rf obj snake cube
