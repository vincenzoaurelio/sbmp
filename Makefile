# NCORES: set to the cores number
# COLOR_CHOSE_FUNC: set to a function with signature
##  static void func(const parallelepiped_t *p, const int *src, float out[3]);
CFLAGS = -Wall -Wextra -DCOLOR_CHOSE_FUNC=average -DNCORES=8 -O2
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
