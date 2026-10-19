# Simple Baboon Manipulation Program
# NCORES: set to the cores number
CFLAGS = -ggdb -Wall -Wextra -DDEBUG -DNCORES=8 -O2
LIBS = -lm -pthread `sdl2-config --cflags --libs` 

all: sbmp
sbmp: main.c img.h
	$(CC) main.c $(LIBS) $(CFLAGS) -o sbmp
clean:
	rm ./sbmp
