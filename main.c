/* Simple Baboon Manipulation Program */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <SDL2/SDL.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <getopt.h>
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
      case 5: r1 = 1; g1 = 0; b1 = x; break;
      default: return NULL;
      }

      int ri = (int)(r1 * 255.0f + 0.5f);
      int gi = (int)(g1 * 255.0f + 0.5f);
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

/* ============================ pipeline =================================== */

typedef enum {
  OP_GREY     = 1 << 0,
  OP_GAUSS    = 1 << 1,
  OP_AVERAGE  = 1 << 2,
  OP_SHARP    = 1 << 3,
  OP_DX       = 1 << 4,
  OP_DY       = 1 << 5,
  OP_MAG      = 1 << 6,
  OP_ORI      = 1 << 7,
  OP_MULSCALAR= 1 << 8,
  OP_SUMSCALAR= 1 << 9,
  OP_SQUARE   = 1 << 10,
  OP_SQRT     = 1 << 11,
  OP_RESCALE  = 1 << 12,
  OP_IMGMIN   = 1 << 13,
  OP_IMGMAX   = 1 << 14,
  OP_SATSUM   = 1 << 15,
  OP_NORMSUM  = 1 << 16,
  OP_SATSUB   = 1 << 17,
  OP_SUB      = 1 << 18,
  OP_MULT     = 1 << 19
} flag_t;

#define ISBINARY(k)						\
  ((k) & (OP_IMGMIN | OP_IMGMAX | OP_SATSUM | OP_NORMSUM |	\
          OP_SATSUB | OP_SUB | OP_MULT))

typedef struct {
  flag_t kind;
  char   *arg;   /* path (binary) or numeric string (G, A, p, a) */
} op_t;

#define MAX_OPS 128

static op_t  ops[MAX_OPS];
static int   nops = 0;
static char *input = NULL;
static char *output = NULL;


static int push_op(flag_t kind, char *arg)
{
  if (nops >= MAX_OPS) {
    fprintf(stderr, "Error: too many operations (max %d)\n", MAX_OPS);
    return 0;
  }
  ops[nops].kind = kind;
  ops[nops].arg  = arg;
  nops++;
  return 1;
}

/* apply a unary operation to img (int**), return a new int** */
static int **apply_unary(int **img, flag_t kind, const char *arg)
{
  int **out = NULL;

  switch (kind) {
  case OP_GREY: {
    float **g = rgb2grey(img);
    if (!g) return NULL;
    out = pack_image(g, g, g);
    freemat(g);
    return out;
  }

  case OP_GAUSS: {
    size_t n = 0;
    float  sigma = 0.0f;
    if (arg) {
      char *end = NULL;
      n = (size_t)strtoul(arg, &end, 10);
      if (end && *end == ':') sigma = strtof(end + 1, NULL);
    }
    if (n == 0) n = 5;
    if (n % 2 == 0) n += 1;
    float **k = gaussker(n, sigma);
    if (!k) return NULL;
    out = rgbconv(img, k);
    freemat(k);
    return out;
  }

  case OP_AVERAGE: {
    size_t n = arg ? (size_t)strtoul(arg, NULL, 10) : 3;
    if (n == 0) n = 3;
    if (n % 2 == 0) n += 1;
    float **k = averageker(n);
    if (!k) return NULL;
    out = rgbconv(img, k);
    freemat(k);
    return out;
  }

  case OP_SHARP: {
    float **k = sharpker();
    if (!k) return NULL;
    out = rgbconv(img, k);
    freemat(k);
    return out;
  }

  case OP_DX: {
    float **k = sobelxker();
    if (!k) return NULL;
    out = rgbconv(img, k);
    freemat(k);
    return out;
  }

  case OP_DY: {
    float **k = sobelyker();
    if (!k) return NULL;
    out = rgbconv(img, k);
    freemat(k);
    return out;
  }

  case OP_MAG: {
    float **g = rgb2grey(img);
    if (!g) return NULL;
    float **m = sobelmag(g);
    freemat(g);
    if (!m) return NULL;
    out = pack_image(m, m, m);
    freemat(m);
    return out;
  }

  case OP_ORI: {
    float **g = rgb2grey(img);
    if (!g) return NULL;
    float **o = sobelori(g);
    freemat(g);
    if (!o) return NULL;
    out = angle2rgb(o);
    freemat(o);
    return out;
  }

  case OP_MULSCALAR: {
    if (!arg) return NULL;
    float k = strtof(arg, NULL);
    float **R = extractred(img);
    float **G = extractgreen(img);
    float **B = extractblu(img);
    mulscalar(R, k); mulscalar(G, k); mulscalar(B, k);
    out = pack_image(R, G, B);
    freemat(R); freemat(G); freemat(B);
    return out;
  }

  case OP_SUMSCALAR: {
    if (!arg) return NULL;
    float k = strtof(arg, NULL);
    float **R = extractred(img);
    float **G = extractgreen(img);
    float **B = extractblu(img);
    sumscalar(R, k); sumscalar(G, k); sumscalar(B, k);
    out = pack_image(R, G, B);
    freemat(R); freemat(G); freemat(B);
    return out;
  }

  case OP_SQUARE: {
    float **R = extractred(img);
    float **G = extractgreen(img);
    float **B = extractblu(img);
    square(R); square(G); square(B);
    out = pack_image(R, G, B);
    freemat(R); freemat(G); freemat(B);
    return out;
  }

  case OP_SQRT: {
    float **R = extractred(img);
    float **G = extractgreen(img);
    float **B = extractblu(img);
    squareroot(R); squareroot(G); squareroot(B);
    out = pack_image(R, G, B);
    freemat(R); freemat(G); freemat(B);
    return out;
  }

  case OP_RESCALE: {
    float **R = extractred(img);
    float **G = extractgreen(img);
    float **B = extractblu(img);
    rescaleu8(R); rescaleu8(G); rescaleu8(B);
    out = pack_image(R, G, B);
    freemat(R); freemat(G); freemat(B);
    return out;
  }

  default:
    return NULL;
  }
}

/* apply a binary operation between A and B (int**), return a new int** */
static int **apply_binary(int **A, int **B, flag_t kind)
{
  if (!A || !B) return NULL;
  if (matrows(A) != matrows(B) || matcols(A) != matcols(B)) {
    fprintf(stderr, "Error: size mismatch in binary op\n");
    return NULL;
  }

  float **RA = extractred(A);
  float **GA = extractgreen(A);
  float **BA = extractblu(A);
  float **RB = extractred(B);
  float **GB = extractgreen(B);
  float **BB = extractblu(B);

  if (!RA || !GA || !BA || !RB || !GB || !BB) {
    freemat(RA); freemat(GA); freemat(BA);
    freemat(RB); freemat(GB); freemat(BB);
    return NULL;
  }

  int **out = NULL;

  switch (kind) {
  case OP_IMGMIN:
    imgmin(RA, RB); imgmin(GA, GB); imgmin(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_IMGMAX:
    imgmax(RA, RB); imgmax(GA, GB); imgmax(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_SATSUM:
    satsum(RA, RB); satsum(GA, GB); satsum(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_NORMSUM:
    normsum(RA, RB); normsum(GA, GB); normsum(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_SATSUB:
    satsubtraction(RA, RB); satsubtraction(GA, GB); satsubtraction(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_SUB:
    subtraction(RA, RB); subtraction(GA, GB); subtraction(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  case OP_MULT:
    mult(RA, RB); mult(GA, GB); mult(BA, BB);
    out = pack_image(RA, GA, BA);
    break;
  default:
    break;
  }

  freemat(RA); freemat(GA); freemat(BA);
  freemat(RB); freemat(GB); freemat(BB);

  return out;
}

/* =============================== main =================================== */

int main(int argc, char **argv)
{
  int **img = NULL;

  const char *optstring =
    "hVvqi:o:dgG:A:SxymOp:a:srRB:T:D:E:F:H:J:";

  int opt;
  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
    case 'o': output = optarg; break;
    case 'g': if (!push_op(OP_GREY, NULL)) goto error; break;
    case 'G': if (!push_op(OP_GAUSS, optarg)) goto error; break;
    case 'A': if (!push_op(OP_AVERAGE, optarg)) goto error; break;
    case 'S': if (!push_op(OP_SHARP, NULL)) goto error; break;
    case 'x': if (!push_op(OP_DX, NULL)) goto error; break;
    case 'y': if (!push_op(OP_DY, NULL)) goto error; break;
    case 'm': if (!push_op(OP_MAG, NULL)) goto error; break;
    case 'O': if (!push_op(OP_ORI, NULL)) goto error; break;
    case 'p': if (!push_op(OP_MULSCALAR, optarg)) goto error; break;
    case 'a': if (!push_op(OP_SUMSCALAR, optarg)) goto error; break;
    case 's': if (!push_op(OP_SQUARE, NULL)) goto error; break;
    case 'r': if (!push_op(OP_SQRT, NULL)) goto error; break;
    case 'R': if (!push_op(OP_RESCALE, NULL)) goto error; break;
    case 'B': if (!push_op(OP_IMGMIN, optarg)) goto error; break;
    case 'T': if (!push_op(OP_IMGMAX, optarg)) goto error; break;
    case 'D': if (!push_op(OP_SATSUM, optarg)) goto error; break;
    case 'E': if (!push_op(OP_NORMSUM, optarg)) goto error; break;
    case 'F': if (!push_op(OP_SATSUB, optarg)) goto error; break;
    case 'H': if (!push_op(OP_SUB, optarg)) goto error; break;
    case 'J': if (!push_op(OP_MULT, optarg)) goto error; break;
    case '?': goto usage;
    case 'h': goto usage;
    }
  }

  /* first positional argument is used as input */
  if (optind < argc) input = argv[optind++];
  else { fprintf(stderr, "Error: missing input file\n"); goto usage; }

  if (optind < argc) {
    fprintf(stderr, "Error: unexpected argument '%s'\n", argv[optind]);
    goto usage;
  }

  img = loadppm(input);
  if (!img) {
    fprintf(stderr, "Error: cannot load '%s'\n", input);
    goto error;
  }

  /* apply operations in order */
  for (int i = 0; i < nops; ++i) {
    int **next = NULL;

    if (ISBINARY(ops[i].kind)) {
      int **B = loadppm(ops[i].arg);
      if (!B) {
        fprintf(stderr, "Error: cannot load '%s'\n", ops[i].arg);
        goto error;
      }
      next = apply_binary(img, B, ops[i].kind);
      freemat(B);
    } else next = apply_unary(img, ops[i].kind, ops[i].arg);


    if (!next) {
      fprintf(stderr, "Error: operation %d failed\n", i);
      goto error;
    }
    freemat(img);

    img = next;

  }

  /* output */
  if (output) {
    FILE *f = fopen(output, "wb");
    if (!f) {
      fprintf(stderr, "Error: cannot open '%s'\n", output);
      goto error;
    }
    size_t r = matrows(img), c = matcols(img);
    fprintf(f, "P6\n%zu %zu\n255\n", c, r);
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < c; ++j) {
        int p = img[i][j];
        fputc((p >> 16) & 0xFF, f);
        fputc((p >>  8) & 0xFF, f);
        fputc( p        & 0xFF, f);
      }
    }
    fclose(f);
  } else visualize(img, "sbmp");
  

  freemat(img);
  return 0;

 usage:
  fprintf(stderr,
	  "Usage: %s INPUT [-o OUTPUT] [options...]\n"
	  "\n"
	  "  -o FILE    output image\n"
	  "  -h         help\n"
	  "\n"
	  "Unary operations:\n"
	  "  -g         grayscale\n"
	  "  -G N[:S]   conv with gaussian kernel N with optional sigma S\n"
	  "  -A N       conv with average kernel N\n"
	  "  -S         conv with sharpening kernel\n"
	  "  -x         conv with Sobel X kernel\n"
	  "  -y         conv with Sobel Y kernel\n"
	  "  -m         Sobel magnitude (on grayscale)\n"
	  "  -O         Sobel orientation (on grayscale, visualized as HSV)\n"
	  "  -p K       multiply by scalar K (saturating)\n"
	  "  -a K       add scalar K (saturating)\n"
	  "  -s         square\n"
	  "  -r         square root\n"
	  "  -R         rescale to [0,255]\n"
	  "\n"
	  "Binary operations:\n"
	  "  -B IMG     min\n"
	  "  -T IMG     max\n"
	  "  -D IMG     saturated sum\n"
	  "  -E IMG     average (normalized sum)\n"
	  "  -F IMG     saturated subtraction\n"
	  "  -H IMG     absolute difference\n"
	  "  -J IMG     multiply (normalized)\n"
	  "\n",
	  argv[0]);
  return 1;

 error:
  freemat(img);
  return 1;
}
