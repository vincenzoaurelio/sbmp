# NCORES: set to the cores number
CFLAGS = -ggdb -Wall -Wextra -DDEBUG -DNCORES=8 -O2
LIBS = -lm -pthread `sdl2-config --cflags --libs` 

all: test
test: main.c
	$(CC) main.c $(LIBS) $(CFLAGS) -o test
clean:
	rm ./test
