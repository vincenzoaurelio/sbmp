# Simple Baboon Manipulation Program
# NCORES: set to the cores number
CFLAGS = -Wall -Wextra  -DNCORES=8 -O2
LIBS = -lm -pthread  

all: sbmp
sbmp: main.c img.h
	$(CC) main.c $(LIBS) `sdl2-config --cflags --libs` $(CFLAGS) -o sbmp

# Compile without graphic interface.
nogui: main.c img.h
	$(CC) main.c $(LIBS) $(CFLAGS) -o sbmp-nogui -DNOGUI

debug: main.c img.h
	$(CC) main.c $(LIBS) `sdl2-config --cflags --libs` $(CFLAGS) -DDEBUG -ggdb -o sbmp-debug

clean:
	rm ./sbmp*
