CFLAGS = -Wall -Wextra -I./include -I./../SDL/include
LDFLAGS = -L./../SDL/lib/x64 -lSDL2

SNAKE_SRCS = src/snake.c src/mat.c src/net.c src/gen.c src/snake_update.c src/snake_render.c src/snake_net.c src/utils.c
SNAKE_OBJS = $(SNAKE_SRCS:src/%.c=obj/%.o)

TEST_SRCS = src/test.cpp src/cube_model.cpp src/cube_utils.cpp
TEST_OBJS = $(TEST_SRCS:src/%.cpp=obj/%.o)

all: snake test

obj:
	mkdir -p obj

snake: $(SNAKE_OBJS)
	gcc $(SNAKE_OBJS) -o snake $(LDFLAGS)

test: $(TEST_OBJS)
	g++ $(TEST_OBJS) -o test $(LDFLAGS)

obj/%.o: src/%.c | obj
	gcc $(CFLAGS) -c $< -o $@

obj/%.o: src/%.cpp | obj
	g++ $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj snake test
