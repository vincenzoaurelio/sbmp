/* Simple Baboon Manipulation Program */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <getopt.h>

#define IMG_IMPLEMENTATION
#include "img.h"

#define PNG_IMPLEMENTATION
#include "png.h"
#define JPEG_IMPLEMENTATION
#include "jpeg.h"

#ifndef NOGUI /* the software can be compiled without sdl */
#include <SDL2/SDL.h>

int visualize(int **img, const char *title)
// TODO maybe visualize hist 
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

#endif


int **angle2rgb(float **ori)
{
  if (!ori) return NULL;

  size_t r = matrows(ori);
  size_t c = matcols(ori);

  int **img = NULL;
  createmat(img, r, c);

  for (size_t i = 0; i < r; ++i) {
    for (size_t j = 0; j < c; ++j) {

      float hp = (ori[i][j] + M_PI) / (2.0f * M_PI) * 6.0f;
      
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
      default: r1 = 1; g1 = 0; b1 = x; break;
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

typedef enum {
  OP_GREY      = 1 << 0,
  OP_GAUSS     = 1 << 1,
  OP_AVERAGE   = 1 << 2,
  OP_SHARP     = 1 << 3,
  OP_DX        = 1 << 4,
  OP_DY        = 1 << 5,
  OP_MAG       = 1 << 6,
  OP_ORI       = 1 << 7,
  OP_MULSCALAR = 1 << 8,
  OP_SUMSCALAR = 1 << 9,
  OP_SQUARE    = 1 << 10,
  OP_SQRT      = 1 << 11,
  OP_STRETCH   = 1 << 12,
  OP_IMGMIN    = 1 << 13,
  OP_IMGMAX    = 1 << 14,
  OP_SATSUM    = 1 << 15,
  OP_NORMSUM   = 1 << 16,
  OP_SATSUB    = 1 << 17,
  OP_SUB       = 1 << 18,
  OP_MULT      = 1 << 19,
  OP_IND       = 1 << 20,
  OP_DITHER    = 1 << 21,
  OP_ERODE     = 1 << 22,
  OP_DILATE    = 1 << 23,
  OP_OPEN      = 1 << 24,
  OP_CLOSE     = 1 << 25,
  OP_EQUALIZE  = 1 << 26,
  OP_RESIZE    = 1 << 27
} flag_t;

#define MAX_OPS 128

#define ISBINARY(k)						\
  ((k) & (OP_IMGMIN | OP_IMGMAX | OP_SATSUM | OP_NORMSUM |	\
          OP_SATSUB | OP_SUB | OP_MULT))

#define ISCOLOR(k)						\
  ((k) & (OP_IND | OP_DITHER | OP_IMGMIN | OP_IMGMAX |		\
          OP_SATSUM | OP_SATSUB | OP_RESIZE))

#define ISLUMA(k) (!ISBINARY(k) && !ISCOLOR(k))

typedef struct {
  flag_t kind;
  char   *arg;
} op_t;

static op_t  ops[MAX_OPS];
static int   nops = 0;
static char *input = NULL;
static char *output = NULL;


float **parse_se(const char *arg)
{
  if (!arg) return NULL;
  if (strncmp(arg, "square:", 7) == 0)  
    return square_se((size_t)strtoul(arg+7, NULL, 10));
  if (strncmp(arg, "disk:",   5) == 0)
    return disk_se  ((size_t)strtoul(arg+5, NULL, 10));
  char *end = NULL;
  size_t n = (size_t)strtoul(arg, &end, 10);
  if (end == arg || *end != '\0' || n == 0) return NULL;
  return square_se(n);
}


static int push_op(flag_t kind, char *arg)
{
  if (nops >= MAX_OPS) {
    fprintf(stderr, "Error: too many operations (max %d)\n", MAX_OPS);
    return 0;
  }
  ops[nops].kind = kind;
  ops[nops].arg = arg;
  nops++;
  return 1;
}

/* Apply a brightness/geometry op to the Y plane of *yuv in place.
 * Returns 1 on success, 0 on failure. Does not touch U/V. */
static int applyunary_luma(yuv_t *yuv, flag_t kind, const char *arg)
{
  if (!yuv || !yuv->Y) return 0;

  switch (kind) {

  case OP_GREY: {
    /* already luma: collapse chroma to zero so yuv2rgb yields grey */
    const size_t r = matrows(yuv->U), c = matcols(yuv->U);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        yuv->U[i][j] = yuv->V[i][j] = 0.0f;
    return 1;
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
    if (!k) return 0;
    float **t = conv(yuv->Y, k);
    freemat(k);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  case OP_AVERAGE: {
    size_t n = arg ? (size_t)strtoul(arg, NULL, 10) : 3;
    if (n == 0) n = 3;
    if (n % 2 == 0) n += 1;
    float **k = averageker(n);
    if (!k) return 0;
    float **t = conv(yuv->Y, k);
    freemat(k);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  case OP_SHARP: {
    float **k = sharpker();
    if (!k) return 0;
    float **t = conv(yuv->Y, k);
    freemat(k);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  case OP_DX: {
    float **k = sobelxker();
    if (!k) return 0;
    float **t = conv(yuv->Y, k);
    freemat(k);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  case OP_DY: {
    float **k = sobelyker();
    if (!k) return 0;
    float **t = conv(yuv->Y, k);
    freemat(k);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  case OP_MAG: {
    float **m = sobelmag(yuv->Y);
    if (!m) return 0;
    freemat(yuv->Y);
    yuv->Y = m;
    /* grey output */
    const size_t r = matrows(yuv->U), c = matcols(yuv->U);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j)
        yuv->U[i][j] = yuv->V[i][j] = 0.0f;
    return 1;
  }

  case OP_ORI: {
    float **o = sobelori(yuv->Y);
    if (!o) return 0;
    int **rgb = angle2rgb(o);
    freemat(o);
    if (!rgb) return 0;
    yuv_t *newy = rgb2yuv(rgb);
    freemat(rgb);
    if (!newy) return 0;
    freemat(yuv->Y); freemat(yuv->U); freemat(yuv->V);
    yuv->Y = newy->Y; yuv->U = newy->U; yuv->V = newy->V;
    free(newy);
    return 1;
  }

  case OP_MULSCALAR:
    if (!arg) return 0;
    mulscalar(yuv->Y, strtof(arg, NULL));
    return 1;

  case OP_SUMSCALAR:
    if (!arg) return 0;
    sumscalar(yuv->Y, strtof(arg, NULL));
    return 1;

  case OP_SQUARE:
    square(yuv->Y);
    return 1;

  case OP_SQRT:
    squareroot(yuv->Y);
    return 1;

  case OP_STRETCH:
    stretch(yuv->Y);
    return 1;

  case OP_EQUALIZE:
    equalize(yuv->Y);
    return 1;

  case OP_ERODE:
  case OP_DILATE:
  case OP_OPEN:
  case OP_CLOSE: {
    if (!arg) return 0;
    float **se = parse_se(arg);
    if (!se) { 
      fprintf(stderr, "Error: bad SE '%s'\n", arg);
      return 0;
    }
    float **t = NULL;
    if      (kind == OP_ERODE)  t = erode (yuv->Y, se);
    else if (kind == OP_DILATE) t = dilate(yuv->Y, se);
    else if (kind == OP_OPEN)   t = opening (yuv->Y, se);
    else                        t = closing(yuv->Y, se);
    freemat(se);
    if (!t) return 0;
    freemat(yuv->Y);
    yuv->Y = t;
    return 1;
  }

  default:
    return 0;
  }
}

static int **applyunary_rgb(int **img, flag_t kind, const char *arg)
{
  switch (kind) {
  case OP_IND: {
    size_t n = arg ? (size_t)strtoul(arg, NULL, 10) : 256;
    ind_t *idx = rgb2ind(img, n);
    if (!idx) return NULL;
    int **out = ind2rgb(idx);
    freeindexed(idx);
    return out;
  }
  case OP_DITHER: {
    size_t n = arg ? (size_t)strtoul(arg, NULL, 10) : 256;
    ind_t *d = dithering(img, n);
    if (!d) return NULL;
    int **out = ind2rgb(d);
    freeindexed(d);
    return out;
  }
  case OP_GREY: {
    /* fallback if called directly on RGB */
    float **g = rgb2grey(img);
    if (!g) return NULL;
    int **out = pack_image(g, g, g);
    freemat(g);
    return out;
  }

  case OP_RESIZE: {
    if (!arg) return NULL;

    char *end = NULL;
    float scale = strtof(arg, &end);
    if (end == arg) {
      fprintf(stderr, "Error: resize scale is not a number: '%s'\n", arg);
      return NULL;
    }

    const char *method = "bilinear";
    if (*end == ':') {
      method = end + 1;
    } else if (*end != '\0') {
      fprintf(stderr, "Error: bad resize argument '%s'\n", arg);
      return NULL;
    }

    if (!(scale > 0.0f)) {
      fprintf(stderr, "Error: resize scale must be > 0\n");
      return NULL;
    }

    if (strcmp(method, "nearest") == 0)  return nearest(img, scale);
    if (strcmp(method, "bilinear") == 0) return bilinear(img, scale);

    fprintf(stderr, "Error: unknown interpolation method '%s' "
                    "(expected 'bilinear' or 'nearest')\n", method);
    return NULL;
  }
  default:
    return NULL;
  }
}

static int **applybinary(int **A, int **B, flag_t kind)
{
  if (!A || !B) return NULL;
  if (matrows(A) != matrows(B) || matcols(A) != matcols(B)) {
    fprintf(stderr, "Error: size mismatch in binary op\n");
    return NULL;
  }


  float **RA, **GA, **BA;
  xchannels(A,RA,GA,BA);
  float **RB, **GB, **BB;
  xchannels(B,RB,GB,BB);

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

static int **intpack(const uint8_t *data,
                               int width, int height, int channels)
{
    int **img = NULL;
    createmat(img, (size_t)height, (size_t)width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const uint8_t *p = data + ((size_t)i * width + j) * channels;
            int r, g, b;

            if (channels == 1) r=g=b = p[0];
            else if (channels==3 || channels==4){
                r=p[0];
                g=p[1];
                b=p[2];
            } else {
                freemat(img);
                return NULL;
            }

            img[i][j] = (r<<16) | (g<<8) | b;
        }
    }
    return img;
}

int **loadppm(const char *path)
{
  int fd = open(path, O_RDONLY);
  if (fd == -1) return NULL;

  uint8_t *buf = NULL;
  int **img = NULL;

  off_t size = lseek(fd, 0, SEEK_END);
  if (size == -1 || lseek(fd, 0, SEEK_SET) == -1) goto panic;

  buf = malloc((size_t) size);
  if (!buf) goto panic;

  if (read(fd, buf, (size_t) size) != size) goto panic;

  /* magic "P6" */
  if (size < 3 || strncmp((const char *) buf, "P6", 2) != 0) goto panic;

  uint8_t *p   = buf + 2;
  uint8_t *end = buf + size;

  size_t vals[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    for (;;) {
      if (p >= end) goto panic;
      if (isspace(*p)) { p++; continue; }
      if (*p == '#') { while (p < end && *p != '\n') p++; continue; }
      break;
    }
    if (!isdigit(*p)) goto panic;
    size_t v = 0;
    while (p < end && isdigit(*p)) v = v * 10 + (size_t)(*p++ - '0');
    vals[i] = v;
  }

  size_t width  = vals[0];
  size_t height = vals[1];

  if (p < end && isspace(*p)) p++;

  if ((size_t)(end - p) < width * height * 3) goto panic;

  createmat(img, height, width);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      int R = *p++;
      int G = *p++;
      int B = *p++;
      img[i][j] = (R << 16) | (G << 8) | B;
    }
  }

  free(buf);
  close(fd);
  return img;

 panic:
  if (buf) free(buf);
  if (img) freemat(img);
  if (fd >= 0) close(fd);
  return NULL;
}

int **loadimg(const char *path)
{
    const char *dot = strrchr(path, '.');
    if (dot) {
        if (strcasecmp(dot+1, "png") == 0) {
            png_image *p = png_load(path);
            if (!p) return NULL;
            int **img = intpack(p->data, p->width, p->height, p->channels);
            png_free(p);
            return img;
        }
        if (strcasecmp(dot+1, "jpg") == 0 ||
            strcasecmp(dot+1, "jpeg") == 0) {
            jpeg_image *j = jpeg_load(path);
            if (!j) return NULL;
            int **img = intpack(j->data, j->width, j->height, j->channels);
            jpeg_free(j);
            return img;
        }
    }
    return loadppm(path); // fallback
}

/* =============================== main =================================== */

int main(int argc, char **argv)
{
  int **img = NULL;

  const char *optstring =
    "hVvqi:o:gG:A:SxymOp:a:srRB:T:D:E:F:H:J:d:e:l:k:c:z:";

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
    case 'R': if (!push_op(OP_STRETCH, NULL)) goto error; break;
    case 'q': if (!push_op(OP_EQUALIZE, NULL)) goto error; break;
    case 'B': if (!push_op(OP_IMGMIN, optarg)) goto error; break;
    case 'T': if (!push_op(OP_IMGMAX, optarg)) goto error; break;
    case 'D': if (!push_op(OP_SATSUM, optarg)) goto error; break;
    case 'E': if (!push_op(OP_NORMSUM, optarg)) goto error; break;
    case 'F': if (!push_op(OP_SATSUB, optarg)) goto error; break;
    case 'H': if (!push_op(OP_SUB, optarg)) goto error; break;
    case 'J': if (!push_op(OP_MULT, optarg)) goto error; break;
    case 'i': if (!push_op(OP_IND, optarg)) goto error; break;
    case 'd': if (!push_op(OP_DITHER, optarg)) goto error; break;
    case 'e': if (!push_op(OP_ERODE,  optarg)) goto error; break;
    case 'l': if (!push_op(OP_DILATE, optarg)) goto error; break;
    case 'k': if (!push_op(OP_OPEN,   optarg)) goto error; break;
    case 'c': if (!push_op(OP_CLOSE,  optarg)) goto error; break;
    case 'z': if (!push_op(OP_RESIZE, optarg)) goto error; break;
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

  img = loadimg(input);
  if (!img) {
    fprintf(stderr, "Error: cannot load '%s'\n", input);
    goto error;
  }

  /* Operations are applied in order.
   *
   * `img` holds the current RGB image. `yuv` is an optional live YUV
   * working copy: it is materialized lazily when the first luma op
   * arrives and kept alive across consecutive luma ops, so a chain like
   * -G 5 -S -R converts to YUV once. It is flushed back to `img` before
   * any color/binary op, before output, and at the end.
   */
  yuv_t *yuv = NULL;

  for (int i = 0; i < nops; ++i) {
    flag_t kind = ops[i].kind;

    if (ISBINARY(kind)) {
      /* flush pending YUV first */
      if (yuv) {
        int **rgb = yuv2rgb(yuv);
        freeyuv(yuv);
        if (!rgb) { fprintf(stderr, "Error: yuv2rgb failed\n"); goto error; }
        freemat(img);
        img = rgb;
      }

      int **B = loadimg(ops[i].arg);
      if (!B) {
        fprintf(stderr, "Error: cannot load '%s'\n", ops[i].arg);
        goto error;
      }
      int **next = applybinary(img, B, kind);
      freemat(B);
      if (!next) {
        fprintf(stderr, "Error: operation %d failed\n", i);
        goto error;
      }
      freemat(img);
      img = next;

    } else if (ISCOLOR(kind)) {
      /* flush pending YUV, then run in RGB */
      if (yuv) {
        int **rgb = yuv2rgb(yuv);
        freeyuv(yuv);
        if (!rgb) { fprintf(stderr, "Error: yuv2rgb failed\n"); goto error; }
        freemat(img);
        img = rgb;
      }
      int **next = applyunary_rgb(img, kind, ops[i].arg);
      if (!next) {
        fprintf(stderr, "Error: operation %d failed\n", i);
        goto error;
      }
      freemat(img);
      img = next;

    } else { /* ISLUMA */
      if (!yuv) {
        yuv = rgb2yuv(img);
        if (!yuv) { fprintf(stderr, "Error: rgb2yuv failed\n"); goto error; }
      }
      if (!applyunary_luma(yuv, kind, ops[i].arg)) {
        fprintf(stderr, "Error: operation %d failed\n", i);
        goto error;
      }
    }
  }

  /* flush any pending YUV before output */
  if (yuv) {
    int **rgb = yuv2rgb(yuv);
    freeyuv(yuv);
    if (!rgb) { fprintf(stderr, "Error: yuv2rgb failed\n"); goto error; }
    freemat(img);
    img = rgb;
  }

  /* output */
  if (output) {
    size_t r = matrows(img), c = matcols(img);

    png_image *p = png_create((int)c, (int)r, 3);
    if (!p) return -1;

    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            int px = img[i][j];
            uint8_t *dst = p->data + (i * c + j) * 3;
            dst[0] = (px >> 16) & 0xFF;
            dst[1] = (px >>  8) & 0xFF;
            dst[2] =  px        & 0xFF;
        }
    }

    if (png_save(p, output) != 0) {
        fprintf(stderr, "Error: writing '%s'\n", output);
        goto error;
    }
  } else {
#ifndef NOGUI
    visualize(img, "sbmp");
#else
    goto usage;
#endif
  }
  

  freemat(img);
  return 0;

 usage:
#ifndef NOGUI
#define OUTPUT "[-o OUTPUT]"
#else
#define OUTPUT "-o OUTPUT"
#endif
  fprintf(stderr,
	  "Usage: %s INPUT "OUTPUT" [options...]\n"
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
	  "  -R         stretch image\n"
	  "  -q         equalize image\n"
	  "  -i N       indexed image with N colors palette\n"
	  "  -d N       dithering with N color palette\n"
      "  -e SE      erosion\n" 
      "             SE: square:N | disk:R | N\n"
      "  -l SE      dilation\n"
      "  -k SE      opening (erode then dilate)\n"
      "  -c SE      closing (dilate then erode)\n"
      "  -z S[:M]   resize by scale S. M: bilinear|nearest, default bilinear\n"
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

/* ===================================BABOON=================================== 
 * ............................................................................
 * .@@......@.......@......@@@@@@.@@@@@@@@@@@..................@.......@.......
 * @......@@..@.@@@@.@@...@@..@@@.............@@@.@@@@@@@@@@@@@@@@@.@@.......@.
 * .@@.@.....@.@@.@@@@@@@@@@@@@@@@@@@.@@@@@@@@@.@@@@..@@@@@@@@@@@@..@......@...
 * ............@@@@@@@@@@...@......@@@@@@@@@@..@@...@@..@...@@......@...@@..@..
 * .@@..@..........@@...@@......@@@.@@@@.@@@..@@@@@...@@@...@....@.@......@@...
 * ...@.........@...@@....@@@@@@@....@.@.@.@@@...@@@@@@...@@@..@@...........@..
 * .@..@.....@..@.@.@.@@@..........@@@.@.@@...@@........@@.@.@.@@..@...........
 * ...@........@...@.....@@@@@@@@@@@@.@@.@.@.@@.@@.@@@@@@@.@...@...........@...
 * ..@@............@@@@..@@@.......@..@@.@@.@@.....@@@..@@.@@...@.@@...@.@...@.
 * .@@..@......@..@.@@@@@@...@@....@@.@@.@@@.@......@@....@.@@@.@@........@.@@.
 * ...@.@..@....@@@@.@@@...@.......@.@@@.@@@.@.....@.@@@..@.....@@..@........@.
 * .@......@...@@.....@@...@@......@.@@@.@@@.@.......@@...@.....@.......@..@.@.
 * .@@.@@@@@@@.......@@.@@..@..@...@.@@@@@@@.@.......@...@@..@.................
 * ...@@.........@@@@@..@....@.@...@.@@@.@@@@@......@...@..@@.@..@..@..@.@..@@.
 * .@@.@..@@...@@@..@@...@@........@@@@@..@@@...........@...@..@@@..@..@.@..@@.
 * .@@.@.....@.....@@..@.@@........@@@@...@@@......@...@....@.@...@@..@@.@@....
 * ....@@.@@..@@..@.@.@@..@........@@@@..@@@@..@......@@....@.@.........@..@@@.
 * .@@@.@@@.@@.@@@@.@..@@..@......@@@@@@.@@@@.........@....@@@...@...@..@.@.@..
 * ...@@@@@@@@@@..@@@@..@.........@@@@@..@@@@........@.....@...@.@.....@.@@.@@.
 * ......@@@@@..@.@.@@..@@..@..@..@@@@@...@@@.@.@...@..@.@.@@.@@@@@@@@@@.@@@@..
 * ..........@@@@.@.@@.@.@.........@@@@...@@@......@....@.@@.@...@.@.@@..@.@@@.
 * ........@@@@@.@@.@@....@..@.....@@@@@.@@@...@.......@..@@@@@@@...@@@@@...@@.
 * .....@..@@@@@@@@@@@@.@.@..@......@@@@@@@@......@..@@..@@@@@@...@.@.@@.@@@...
 * .......@....@@@@@..@@...@.......@@@@..@@@..@..@..@@..@@........@@@.@...@@@@.
 * ....@@@...@@.@@..@@@@.@@.@.@....@@@@.@@@@..@..@.@@.@@.@@@@@@@@@..@.@....@@@.
 * ........@@@@@@@@..@@@@...@.@...@@@@@.@@@@....@....@.@@.@@...@@@@@@@.@@@@@.@.
 * ...........@.....@@..@@.@....@@@@@@@.@@@@@......@@@@@...@@@@@..@@@@@@@@@.@@.
 * .......@@.@@......@@@@@.@@@@@@@@@@@@@@@@@@@@@@...@...@@@..........@@@@@@@.@.
 * .........@@........@...@@@@@@....@@@@@.......@@@@.@.@...@@@.@@@@@@..@@.@@.@.
 * .........@.@.......@@.@@@@@..@@@@@.....@@@@.@@@@@...@@@.@.@@.....@@@.@.@@...
 * ........@@@........@.@@@@@@...@@@@@@@.@@@@@@@@@@@@.@...@@@@.@@@@........@@..
 * .........@..@@@......@..@@@@@@@@@@...@@..@@@@@@@.@@@@@@.@...........@@@.....
 * .@@.....@@.@............@@@@@@@@@@@@@@@@@@@@@@@@...@@....@@.......@@@@......
 * ...........@.@@.@..........@@@@@@@@@@@@@@@@@@@@@@......@@............@@.....
 * ....@....@.@@@...............@.@.@.@@@@@@@@...........@..@....@@.@@@..@..@..
 * ..........@..@.@@.......@@.........................@...@@.@.......@@@@.@..@.
 * ......@@...@..@..@@...........@@@@@@@.@@@.@.@.@.@@@.@.@...........@@.@.@....
 * ...........@.@@.@@..@.........@@@@@@@.@@@.@@.@..@@@.....@.@.@.@@@.@@.@@..@..
 * ..@............@..@.......................@................@....@.@@...@..@@
 * ...........................................................@@.@..@......@.*/
