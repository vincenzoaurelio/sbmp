#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <SDL2/SDL.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#define IMG_IMPLEMENTATION
#include "img.h"

int visualize(int **img, const char *title)
{
  if (!img) return 0;

  int width  = (int) matcols(img);
  int height = (int) matrows(img);

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Window *win =
    SDL_CreateWindow(title,
                     SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                     width, height, SDL_WINDOW_SHOWN);

  SDL_Surface *surface = SDL_GetWindowSurface(win);

  memset((uint8_t *) surface->pixels, 0, surface->pitch * surface->h);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint32_t p = (uint32_t) img[y][x];
      uint8_t r = (uint8_t)((p >> 16) & 0xFF);
      uint8_t g = (uint8_t)((p >>  8) & 0xFF);
      uint8_t b = (uint8_t)( p        & 0xFF);
      ((uint8_t *)surface->pixels)[y * surface->pitch + x*4 + 0] = b;
      ((uint8_t *)surface->pixels)[y * surface->pitch + x*4 + 1] = g;
      ((uint8_t *)surface->pixels)[y * surface->pitch + x*4 + 2] = r;
      ((uint8_t *)surface->pixels)[y * surface->pitch + x*4 + 3] = 255;
    }
  }

  int running = 1;
  while (running) {
    SDL_UpdateWindowSurface(win);

    SDL_Event event;
    while (SDL_PollEvent(&event))
      if (event.type == SDL_QUIT) running = 0;

    SDL_Delay(16);
  }

  SDL_DestroyWindow(win);
  SDL_Quit();

  return 1;
}

static int **rgbconv(int **img, float **k)
{
  if (!img || !k) return NULL;

  float **R = extractred(img);
  float **G = extractgreen(img);
  float **B = extractblu(img);

  float **Rc = conv(R, k);
  float **Gc = conv(G, k);
  float **Bc = conv(B, k);

  int **out = pack_image(Rc, Gc, Bc);

  freemat(R);  freemat(G);  freemat(B);
  freemat(Rc); freemat(Gc); freemat(Bc);

  return out;
}
/* Convert a matrix of radians in (-pi, pi] into a packed RGB image.
 * Returns a new int** of the same shape as ori, or NULL on failure. */
int **angle2rgb(float **ori)
{
  if (!ori) return NULL;

  size_t r = matrows(ori);
  size_t c = matcols(ori);

  int **img = NULL;
  createmat(img, r, c);

  for (size_t i = 0; i < r; ++i) {
    for (size_t j = 0; j < c; ++j) {

      float hp = 3.0f * (ori[i][j] + M_PI) / M_PI;
      hp = fmodf(hp, 6.0f);
      if (hp < 0.0f) hp += 6.0f;

      float x  = 1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f);

      float r1, g1, b1;
      switch ((int)hp) {
        case 0: r1 = 1; g1 = x; b1 = 0; break;
        case 1: r1 = x; g1 = 1; b1 = 0; break;
        case 2: r1 = 0; g1 = 1; b1 = x; break;
        case 3: r1 = 0; g1 = x; b1 = 1; break;
        case 4: r1 = x; g1 = 0; b1 = 1; break;
        case 5: r1 = 1; g1 = 0; b1 = x; break;   /* hp == 5 */
	default: return NULL;
      }

      int ri = (int)(r1 * 255.0f + 0.5f);
      int gi = (int)(g1 * 255.0f + 0.5f); /* round */
      int bi = (int)(b1 * 255.0f + 0.5f);

      if (ri < 0) ri = 0; 
      if (gi < 0) gi = 0; 
      if (bi < 0) bi = 0; 

      if (ri > 255) ri = 255;
      if (gi > 255) gi = 255;
      if (bi > 255) bi = 255;
      
      img[i][j] = (ri << 16) | (gi << 8) | bi;
    }
  }
  return img;
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
  int   **img     = NULL;
  int   **out     = NULL;
  int   **vis     = NULL;
  float **k       = NULL;
  float **gray    = NULL;
  float **mag     = NULL;
  float **ori     = NULL;

  size_t n;
  float sigma = 0;
  const char *path;

  if (argc < 2) goto usage;

  path = argv[1];
  n    = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10) : 5;

  if (n == 0) n = 1;
  if (n % 2 == 0) n += 1;   /* conv needs uneven kernel */

  img = loadppm(path);
  if (!img) goto error;

  visualize(img, "original");

  k = averageker(n);
  if (!k) goto error;
  out = rgbconv(img, k);
  if (!out) goto error;
  visualize(out, "averageker");
  freemat(out); freemat(k);

  k = gaussker(n, sigma);
  if (!k) goto error;
  out = rgbconv(img, k);
  if (!out) goto error;
  visualize(out, "gaussker");
  freemat(out); freemat(k);

  k = sharpker();
  if (!k) goto error;
  out = rgbconv(img, k);
  if (!out) goto error;
  visualize(out, "sharpker");
  freemat(out); freemat(k);

  /* ---------- Sobel ---------- */

  k = sobelxker();
  if (!k) goto error;
  out = rgbconv(img, k);
  if (!out) goto error;
  visualize(out, "sobelxker");
  freemat(out); freemat(k);

  k = sobelyker();
  if (!k) goto error;
  out = rgbconv(img, k);
  if (!out) goto error;
  visualize(out, "sobelyker");
  freemat(out); freemat(k);

  gray = rgb2grey(img);
  if (!gray) goto error;

  mag = sobelmag(gray);
  if (!mag) goto error;

  vis = pack_image(mag, mag, mag);
  if (!vis) goto error;
  visualize(vis, "sobelmagnitude");
  freemat(vis);

  ori = sobelori(gray);
  if (!ori) goto error;
  int **i;
  i=angle2rgb(ori);
  if (!i) goto error;
  visualize(i, "sobelorientation");
  freemat(i);

  freemat(ori);
  freemat(mag);
  freemat(gray);
  freemat(out);
  freemat(k);
  freemat(img);
  return 0;

 usage:
  fprintf(stderr, "Usage: %s <image.ppm> [n]\n", argv[0]);
  return 1;

 error:
  freemat(ori);
  freemat(mag);
  freemat(gray);
  freemat(out);
  freemat(k);
  freemat(vis);
  freemat(img);
  return 1;
}

