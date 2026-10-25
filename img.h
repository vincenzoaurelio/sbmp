#ifndef IMG_H
#define IMG_H

#include <stddef.h>

/* ================================macros=================================== */
#define EPS 1e-3

#ifndef NCORES     // number of cores
#define NCORES 4 
#endif

/* empirical thresholds */
#define MULTITHREAD_THRESHOLD_MATMAT     2500   /* matrows * matcols */
#define MULTITHREAD_THRESHOLD_MATVEC   250000   /* matrows * matcols */
#define MULTITHREAD_THRESHOLD_VECVEC   100000   /* vecsize           */
/* when R*C*P*Q > N^2 * log2(N) FFT is better
 * N is the next power of two of max(R+P-1, C+Q-1)).
 * Is used an empirical threshold on the number of direct operations to avoid
 * to compute nextpow2 every call of conv */
#define FFT_THRESHOLD_CONV  (1u << 22) 

/* =========================================================================
 * Create special matrices and vectors of arbitrary type with dimentions
 * inside the header.
 * The object parsed to the other functions of the library is a pointer or a
 * pointer to a pointer that hides informations behind the head of the array.
 *
 * +----------+-------------///
 * |size_t dim|float *vec ...
 * +----------+-------------///
 *
 * +----------+-------------///
 * |size_t dim|int *vec ...
 * +----------+-------------///
 *
 * +----------+----------+--------------///
 * | size_t r | size_t c |char **mat ...
 * +----------+----------+^-------------///
 *                        |
 *                     pointer returned by the functions
 * 
 * vecsize(vec), matrows(mat), matcols(mat) macros can be used to access to the
 * header's data.
 * ========================================================================= */

#define createvec(v,dim)     					\
  do {								\
    header_t *h = xmalloc(sizeof(*(v))*(dim) + sizeof(*h));	\
    h->r=(dim);							\
    h->c=1;							\
    v = (typeof(v))(h + 1);					\
  } while (0)

#define createmat(m,rr,cc)						\
  do {									\
    header_t *_h = xmalloc(sizeof(**(m))*(rr)*(cc)+			\
			   (rr)*sizeof(*(m))+sizeof(*_h));		\
    _h->r=rr;								\
    _h->c=cc;								\
    m = (typeof(m))(_h+1);						\
    typeof(**(m)) *data=(typeof(**m)*) (m + rr);			\
    for (size_t _i = 0; _i < (rr); ++_i) m[_i] = data + _i * (cc);	\
  } while(0)

#define zeros(m,rr,cc)							\
  do {									\
    header_t *_h = xcalloc(1,sizeof(**(m))*(rr)*(cc)+			\
			   (rr)*sizeof(*(m))+sizeof(*_h));		\
    _h->r=rr;								\
    _h->c=cc;								\
    m = (typeof(m))(_h+1);						\
    typeof(**(m)) *data=(typeof(**m)*) (m + rr);			\
    for (size_t _i = 0; _i < (rr); ++_i) m[_i] = data + _i * (cc);	\
  } while(0)

#define getheader(x) ((x) ? (header_t*)(x) - 1 : NULL)

#define freemat(m)				\
  do {						\
    if (m) {					\
      free(getheader(m));			\
      m = NULL;					\
    }						\
  } while(0)

#define freevec(v) freemat(v)

#define vecsize(vec) ((vec) ? (getheader(vec) -> r) : 0)
#define matrows(mat) ((mat) ? (getheader(mat) -> r) : 0)
#define matcols(mat) ((mat) ? (getheader(mat) -> c) : 0)


/* ================================structs================================== */
typedef struct
{
  size_t r;   // rows 
  size_t c;   // columns
} header_t;   // heaader for both matrices and vectors

typedef struct
{
  size_t n;         
  uint16_t **i; // indexed image 
  float (*palette)[3]; // n x 3 
} indexed_t;

/* -----------------------------multithread--------------------------------- */
typedef struct
{
  size_t start, end;
  float **A, **Bt;
  float **result;
} multiply_t;

typedef struct
{
  size_t start, end;
  float **A;
  float *v;
  float *result; 
} matvec_t;

typedef struct
{
  size_t start, end;
  float r;
  float *a, *b;
} dot_t;

/* =========================function signatures============================= */
static inline void *xmalloc(size_t size);
static inline void *xcalloc(size_t n, size_t size);

#ifdef DEBUG
/* -------------fill with random numbers in a [min, max] interval----------- */
float *randvec(size_t l, int max);
float **randmat(size_t r, size_t c, int max);
#endif

/* -------------------------print data structures--------------------------- */
#define printv(vec)							\
  do {									\
    printf("{");							\
    for (size_t _i=0; _i<vecsize(vec); ++_i){				\
      _Generic((vec)[0],						\
	       int: printf("%d%s",(vec)[_i],				\
			   (_i!=vecsize(vec)-1)?", ":""),		\
	       float: printf("%f%s",(vec)[_i],				\
			     (_i!=vecsize(vec)-1)?", ":""),		\
	       double: printf("%lf%s",(vec)[_i],			\
			      (_i!=vecsize(vec)-1)?", ":""),		\
	       char: printf("%c%s",(vec)[_i],				\
			    (_i!=vecsize(vec)-1)?", ":""),		\
	       long: printf("%ld%s",(vec)[_i],				\
			    (_i!=vecsize(vec)-1)?", ":""),		\
	       unsigned int: printf("%u%s",(vec)[_i],			\
				    (_i!=vecsize(vec)-1)?", ":""),	\
	       default: printf("(unsupported type)"));			\
    }									\
    printf("}\n");							\
  } while(0)

#define printm(mat)							\
  do {									\
    printf("{\n");							\
    for (size_t _i=0;_i<matrows(mat);++_i){				\
      printf("  {");							\
      for (size_t _j=0;_j<matcols(mat);++_j){				\
	_Generic((mat)[0][0],						\
		 int: printf("%d%s",(mat)[_i][_j],			\
			     (_j!=matcols(mat)-1)?", ":""),		\
		 float: printf("%f%s",(mat)[_i][_j],			\
			       (_j!=matcols(mat)-1)?", ":""),		\
		 double: printf("%lf%s",(mat)[_i][_j],			\
				(_j!=matcols(mat)-1)?", ":""),		\
		 char: printf("%c%s",(mat)[_i][_j],			\
			      (_j!=matcols(mat)-1)?", ":""),		\
		 long: printf("%ld%s",(mat)[_i][_j],			\
			      (_j!=matcols(mat)-1)?", ":""),		\
		 unsigned int: printf("%u%s",(mat)[_i][_j],		\
				      (_j!=matcols(mat)-1)?", ":""),	\
		 default: printf("(unsupported type)"));		\
      }									\
      printf("}\n");							\
    }									\
    printf("}\n");							\
  } while (0)


static inline double eps(void); // smallest value for the machine

/* ===========================simple operations============================= */
#define castvec(v,type)							\
  ({									\
    type *_v;								\
    createvec(_v,vecsize(v));						\
    for(size_t _i=0; _i<vecsize(v); ++_i) _v[_i] = (type)(v)[_i];	\
    _v;									\
  })

#define castmat(m,type)				\
  ({						\
    type **_m;					\
    createmat(_m, matrows(m), matcols(m));	\
    for(size_t _i=0; _i<matrows(m); ++_i)	\
      for(size_t _j=0; _j<matcols(m); ++_j)	\
	_m[_i][_j] = (type)(m)[_i][_j];		\
    _m;						\
  })

/* copies src into dest */
#define veccpy(dest,src)			\
  do {						\
    size_t _d=vecsize(dest);			\
    if (vecsize(src) <= _d){			\
      for(size_t _i=0; _i<_d; ++_i)		\
	(dest)[_i]=(src)[_i];			\
    }						\
  } while(0)

#define matcpy(dest,src)				\
  do {							\
    const size_t _r = matrows(dest), _c=matcols(dest);	\
    if (matrows(src) <= _r && matcols(src) <= _c){	\
      for(size_t _i=0; _i<_r; ++_i)			\
	for(size_t _j=0; _j<_c; ++_j)			\
	  (dest)[_i][_j]=(src)[_i][_j];			\
    }							\
  } while(0)

/* returns n if the matrix is square, 0 otherwise */
#define issquare(mat) (matrows(mat)==matcols(mat)) ? matcols(mat) : 0

/* creates an identity matrix d x d */
#define identity(m,d)				\
  do {						\
    createmat(m,d,d);				\
    for (size_t _i=0; _i<(d); ++_i)		\
      for (size_t _j=0; _j<(d); ++_j)		\
	(m)[_i][_j] = (_i == _j) ? 1 : 0;	\
  } while(0)

#define transpose(mat)				\
  ({						\
    size_t _r=matrows(mat), _c=matcols(mat);	\
    typeof(mat) _t= NULL;			\
    createmat(_t, _c, _r);			\
    for(size_t _i=0; _i<_r; ++_i)		\
      for(size_t _j=0; _j<_c; ++_j)		\
	_t[_j][_i]=(mat)[_i][_j];		\
    _t;						\
  })						


#ifdef DEBUG
float **multiply_sequential(float **mat1, float **mat2);
float **multiply_multithread(float **mat1, float **mat2);
float *multiplymatvec_sequential(float **mat, float *vec);
float *multiplymatvec_multithread(float **mat, float *vec);
float dotproduct_multithread(float *a, float *b);
float dotproduct_sequential(float *a, float *b);
#endif

float **multiply(float **A, float **B);
float *multiplymatvec(float **mat, float *vec);
float dotproduct(float *a, float *b);
float **mpow(float **mat, char exponent); // matrix power

/* returns the smallest element in a matrix */
#define matmin(mat)					\
  ({							\
    float _m = INFINITY;				\
    for (size_t _i = 0; _i < matrows(mat); ++_i)	\
      for (size_t _j = 0; _j < matcols(mat); ++_j)	\
        _m = fminf(_m, (mat)[_i][_j]);			\
    _m;							\
  })

/* returns the greatest element in a matrix */
#define matmax(mat)					\
  ({							\
    float _m = -INFINITY;				\
    for (size_t _i = 0; _i < matrows(mat); ++_i)	\
      for (size_t _j = 0; _j < matcols(mat); ++_j)	\
        _m = fmaxf(_m, (mat)[_i][_j]);			\
    _m;							\
  })

/* ===================================img=================================== */

/* Read a binary PPM (P6) into an int** */
int **loadppm(const char *path);

/* Convert a packed-RGB int image to float grayscale using weights
 * Y = 0.299*R + 0.587*G + 0.114*B.
 * Returns a float** with one channel per pixel. */
float **rgb2grey(int **img);
int **pack_image(float **R, float **G, float **B);
void rescaleu8(float **img);

/* ------------------------------ convolution ------------------------------ */
#ifdef DEBUG
float **conv_standard(float **m, float **k);
float **conv_fft(float **m, float **k);
float **conv_lut(float **m, float **k);
/* For conv_lut                                                ***enjambement*
 * Actually the overhead is more than the gain starting to PxQ=21, so it's
 * more convenient to use an hardcoded conv that only works for 3x3 kernels,
 * thats how is done inside conv_standad implementation */
#endif
float **conv(float **m, float **k);
/* Gaussian kernel of size n x n. If sigma <= 0, is used the OpenCV heuristic
 * for choosing a Gaussian standard deviation when only the kernel size n is
 * known. */
float **gaussker(size_t n, float sigma);
/* 3x3 sharpening kernel  */
float **sharpker(void);
/* Average kernel of size nxn. Every element is 1/(n*n). */
float **averageker(size_t n);
float **sobelxker(void);
float **sobelyker(void);
/* Sobel gradient magnitude.
 * dx = conv(I, Kx),   dy = conv(I, Ky)
 * sobelmag(I) = sqrt(dx.^2 + dy.^2)
 * returns a float** of the same shape as I, or NULL on failure. */
float **sobelmag(float **img);
/* Sobel gradient orientation.
 * sobelori(I) = atan(dy/dx)
 * returns a float** of the same shape as I, or NULL on failure. */
float **sobelori(float **img);

/* -----------------------quantization and dithering------------------------ */

/* rgb2ind uses Heckbert median cut color quantization.
 * It takes a packed rgb image and the palette size n, that must be greater
 * than zero, and returns an indexed_t* or NULL on failure. 
 * COLOR_CHOSE_FUNC defines the function used by rgb2ind to chose a color for
 * each parallelepiped. By default it's value is average */
#ifndef COLOR_CHOSE_FUNC
#define COLOR_CHOSE_FUNC(p,src,out) average(p,src,out)
#endif
indexed_t *rgb2ind(int **img, size_t n);
int **ind2rgb(indexed_t *img);
/* Floyd Steinberg error diffusion.
 * The palette is built from img with Heckbert's algorithm by rgb2ind
 * The Floyd Steinberg weights are (7/16, 3/16, 5/16, 1/16).
 * Returns indexed_t* or NULL on failure. */
indexed_t *dithering(int **img, size_t n);
/* freeindexed to free indexed_t */
#define freeindexed(ind)			\
  do {						\
    if (ind) {					\
      freemat(ind->i);				\
      free(ind->palette);			\
      free(ind);				\
      ind = NULL;				\
    }						\
  } while(0)

/* ----------------------Artithmetic and logic operators-------------------- */

#define min(i,j) (((i) > (j)) ? (j) : (i))
#define max(i,j) (((i) < (j)) ? (j) : (i))

#define xchannels(img, R, G, B)					\
  do{								\
    if (img) {							\
      const size_t _r = matrows(img), _c = matcols(img);	\
      createmat(R, _r, _c);					\
      createmat(G, _r, _c);					\
      createmat(B, _r, _c);					\
      for (size_t _i = 0; _i < _r; ++_i){			\
        for (size_t _j = 0; _j < _c; ++_j){			\
          R[_i][_j] = (float)(((img)[_i][_j] >> 16) & 0xFF);	\
	  G[_i][_j] = (float)(((img)[_i][_j] >> 8) & 0xFF);	\
          B[_i][_j] = (float)((img)[_i][_j] & 0xFF);		\
	}							\
      }								\
    }								\
  } while(0)


/* saturation to 255 */
#define mulscalar(I,k)					\
  do {							\
    const size_t _r = matrows(I), _c = matcols(I);		\
    if ((k) >= 0) {					\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min(255, (I)[_i][_j] * (k));	\
    }							\
  } while (0)

/* saturation to 255 */
#define sumscalar(I,k)					\
  do {							\
    const size_t _r = matrows(I), _c = matcols(I);		\
    if ((k) >= 0) {					\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min(255, (I)[_i][_j] + (k));	\
    }							\
  } while (0)


#define imgmin(I,J)					\
  do {							\
    const size_t _r = matrows(I), _c = matcols(I);		\
    if (matrows(J) == _r && matcols(J) == _c) {		\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min((I)[_i][_j], (J)[_i][_j]);	\
    }							\
  } while (0)


#define imgmax(I,J)					\
  do {							\
    const size_t _r = matrows(I), _c = matcols(I);		\
    if (matrows(J) == _r && matcols(J) == _c) {		\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = max((I)[_i][_j], (J)[_i][_j]);	\
    }							\
  } while (0)

/* saturation to 255 */
#define satsum(I,J)						\
  do {								\
    const size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = min(255, (I)[_i][_j] + (J)[_i][_j]);	\
    }								\
  } while (0)

/* normalized in [0, 255] */
#define normsum(I,J)						\
  do {								\
    const size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = ((I)[_i][_j] + (J)[_i][_j]) / 2;	\
    }								\
  } while (0)

/* saturation to 255 */
#define satsubtraction(I,J)					\
  do {								\
    const size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = max(0, (I)[_i][_j] - (J)[_i][_j]);	\
    }								\
  } while (0)


#define subtraction(I,J)					\
  do {								\
    const size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = fabs((I)[_i][_j] - (J)[_i][_j]);	\
    }								\
  } while (0)

/* normalized in [0, 255]  */
#define mult(I,J)						\
  do {								\
    const size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = (I)[_i][_j] * (J)[_i][_j] / 255;	\
    }								\
  } while (0)

/* normalized in [0, 255]  */
#define square(I) mult(I,I)

/* normalized in [0, 255] */
#define squareroot(I)				\
  do {						\
    const size_t _r = matrows(I), _c = matcols(I);	\
    for (size_t _i = 0; _i < _r; ++_i)		\
      for (size_t _j = 0; _j < _c; ++_j)	\
        (I)[_i][_j] = sqrtf((I)[_i][_j]);	\
  } while (0)

#ifdef IMG_IMPLEMENTATION
#undef IMG_IMPLEMENTATION

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>


inline void *xmalloc(size_t size) //malloc wrapper
{
  void *m = malloc(size);
  if(!m) {
    fprintf(stderr,"Error: memory allocation error\n");
    exit(1);
  } return m;
}

inline void *xcalloc(size_t n, size_t size) //calloc wrapper
{
  void *m = calloc(n,size);
  if(!m) {
    fprintf(stderr,"Error: memory allocation error\n");
    exit(1);
  } return m;
}


/* Cleve Moler Algorithm */
static inline double eps(void)
{
  float a = 4.0/3.0;
  float b = a-1;
  float c = b+b+b;
  return 1-c;
}


float *randvec(size_t l, int max)
{
  float *v = NULL;
  createvec(v,l);
  /* generates a random float in the [0,max] interval */
  for(size_t i=0;i<l;++i) v[i] = (float)rand()/RAND_MAX*(float)(max); 
    						    
  return v;					    
}

float **randmat(size_t r, size_t c, int max)
{
  float **m = NULL;
  createmat(m,r,c);
  for (size_t i=0; i<r; ++i)
    for(size_t j=0; j<c; ++j)               
      m[i][j] = (float)rand()/RAND_MAX*(float)(max);

  return m;
}

float dotproduct_sequential(float *a, float *b)
{
  size_t n=vecsize(a);
  if (n!=vecsize(b)) return NAN;
  float sum = 0;
  for (size_t i = 0; i<n; ++i) sum+=a[i]*b[i];
  return sum;
}

void *dotproductele(void *arg)
{
  dot_t *data = (dot_t *)arg;
  float r=0;
  for (size_t i=data->start; i<data->end; ++i) r += data->a[i]*data->b[i];

  data->r = r;  
  return NULL;
}

float dotproduct_multithread(float *a, float *b)
{
  size_t n=vecsize(a);
  if (n!=vecsize(b)) return NAN;
  
  pthread_t threads[NCORES];
  dot_t data[NCORES];

  size_t el_per_thread = n / NCORES;
      
  for (int i=0; i<NCORES; ++i){
    data[i].a = a;
    data[i].b = b;
    data[i].r = 0;
    data[i].start = i * el_per_thread;
    data[i].end = (i==NCORES-1) ? n : (i+1)*el_per_thread;

    pthread_create(&threads[i], NULL, dotproductele, &data[i]);
  }

  float sum=0;
  for (int i = 0; i<NCORES; ++i){
    pthread_join(threads[i], NULL);
    sum+=data[i].r;
  }
  
  return sum;
}


/* returns the multiply between A and B or a  NULL pointer on failure */
float **multiply_sequential(float **A, float **B)
{
  if (matcols(A) != matrows(B)) return NULL;
  
  float **product= NULL;
  createmat(product, matrows(A),matcols(B));
  float **Bt=transpose(B);
  
  for (size_t i=0; i<matrows(A); ++i){
    for (size_t j=0; j<matcols(B); ++j){
      product[i][j]=0;
      for (size_t k=0; k<matcols(A); ++k)
	product[i][j]+=A[i][k]*Bt[j][k];
    }
  }
  freemat(Bt);
  
  return product;
}

void *multiplyrow(void *arg)
{
  multiply_t* data = (multiply_t*)arg;
  
  for (size_t i = data->start; i<data->end; ++i) {
    for (size_t j = 0; j<matrows(data->A); ++j){
      float r = 0;
      for (size_t k = 0; k<matcols(data->A); ++k)
	r += data->A[i][k] * data->Bt[j][k];
      data->result[i][j] = r;
    }
  }
  return NULL;
}

float **multiply_multithread(float **A, float **B)
{
  if (matcols(A) != matrows(B)) return NULL;

  pthread_t threads[NCORES];
  multiply_t data[NCORES];

  float **result= NULL;
  createmat(result, matrows(A),matcols(B));
  float **Bt=transpose(B);
    
  int rows_per_thread = matrows(A) / NCORES;
  for (int i=0; i<NCORES; ++i) {
    data[i].A = A;
    data[i].Bt = Bt;
    data[i].result = result;
    data[i].start =i* rows_per_thread;
    
    data[i].end =
      (i == NCORES-1) ?
      matrows(A) :
      (size_t)(i+1) * rows_per_thread;
    
    pthread_create(&threads[i], NULL, multiplyrow, &data[i]);
  }
  
  for (int i = 0; i<NCORES; i++) pthread_join(threads[i], NULL);

  freemat(Bt);
  return result;
}

float *multiplymatvec_sequential(float **mat, float *vec)
{
  const size_t m = matrows(mat), n = matcols(mat);
  if (n != vecsize(vec)) return NULL;

  float *product =  NULL;
  createvec(product, m);
    
  for (size_t i = 0; i<m; ++i) {
    product[i] = 0;
    for (size_t j = 0; j<n; ++j)
      product[i]+=mat[i][j] * vec[j];
  }
    
  return product;
}

void *multiplymatvecrow(void *arg)
{
  matvec_t *data = (matvec_t *)arg;
  size_t n = vecsize(data->v);
  for (size_t i = data->start; i < data->end; ++i) {
    float r = 0;
    for (size_t j = 0; j<n; ++j)
      r+=data->A[i][j] * data->v[j];
    data->result[i] = r;
  }
  return NULL; 
}


float *multiplymatvec_multithread(float **A, float *v)
{
  size_t m = matrows(A);
  size_t n = matcols(A);
  if (n != vecsize(v)) return NULL;

  pthread_t threads[NCORES];
  matvec_t data[NCORES];

  float *result = NULL;
  createvec(result, m);
    
  int rows_per_thread = matrows(A) / NCORES;
  for (int i=0; i<NCORES; ++i) {
    data[i].A = A;
    data[i].v = v;
    data[i].result = result;
    data[i].start =i* rows_per_thread;
    
    data[i].end =
      (i == NCORES-1) ?
      matrows(A) :
      (size_t)(i+1) * rows_per_thread;
    
    if (pthread_create(&threads[i], NULL, multiplymatvecrow, &data[i]))
      return NULL;
  }
  
  for (int i = 0; i<NCORES; i++)
    if (pthread_join(threads[i], NULL)) return NULL; 

  return result;  
}


float **multiply(float **A, float **B)
{
  return matrows(A)*matcols(A) < MULTITHREAD_THRESHOLD_MATMAT ?
    multiply_sequential(A, B) :
    multiply_multithread(A, B);
}

float *multiplymatvec(float **A, float *v)
{
  return matrows(A)*matcols(A) < MULTITHREAD_THRESHOLD_MATVEC ?
    multiplymatvec_sequential(A, v) :
    multiplymatvec_multithread(A, v);
}

float dotproduct(float *a, float *b)
{
  return vecsize(a) < MULTITHREAD_THRESHOLD_VECVEC ?
    dotproduct_sequential(a, b):
    dotproduct_multithread(a,b);
}

float **mpow(float **mat, char exponent)
{
  size_t c = issquare(mat);
  if(!c) return NULL;
  if (exponent == 0) {
    float **i;
    identity(i,c);
    return i;
  }
  
  float **pow= NULL;
  createmat(pow, c,c);
  matcpy(pow, mat);
  for (char i=1; i<exponent; ++i){
    float **tmp=multiply(pow,mat);
    if (!tmp) {
      freemat(pow);
      return NULL;
    }
    freemat(pow);
    pow=tmp;
  }
 
  return pow;
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

float **rgb2grey(int **img)
{
  float **R=NULL, **G=NULL, **B=NULL;
  xchannels(img,R,G,B);

  const size_t r = matrows(img), c = matcols(img);
  float **grey = NULL;
  createmat(grey, r, c);

  for (size_t i = 0; i < r; ++i)
    for (size_t j = 0; j < c; ++j)
      grey[i][j] = 0.299f * R[i][j] + 0.587f * G[i][j] + 0.114f * B[i][j];

  freemat(R);
  freemat(G);
  freemat(B);
  return grey;
}

int **pack_image(float **R, float **G, float **B)
{
  if (!R || !G || !B) return NULL;

  const size_t r = matrows(R), c = matcols(R);
  if (matrows(G) != r || matcols(G) != c) return NULL;
  if (matrows(B) != r || matcols(B) != c) return NULL;

  int **img = NULL;
  createmat(img, r, c);

  for (size_t i = 0; i < r; ++i) {
    for (size_t j = 0; j < c; ++j) {
      /* limit values to [0,255] then round to nearest int */
      int ri = (int)lrintf(fminf(255.0f, fmaxf(0.0f, R[i][j])));
      int gi = (int)lrintf(fminf(255.0f, fmaxf(0.0f, G[i][j])));
      int bi = (int)lrintf(fminf(255.0f, fmaxf(0.0f, B[i][j])));

      img[i][j] = (ri << 16) | (gi << 8) | bi;
    }
  }
  return img;
}

float **gaussker(size_t n, float sigma)
{
  if (n == 0) return NULL;

  if (sigma <= 0.0f)
    sigma = 0.3f * ((float)n - 1.0f) * 0.5f - 1.0f + 0.8f;

  float **k = NULL;
  zeros(k, n, n);

  const float center = ((float)n - 1.0f) * 0.5f;
  const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
  float sum = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float di = (float)i - center;
      float dj = (float)j - center;
      float v  = expf(-(di * di + dj * dj) * inv2s2);
      k[i][j] = v;
      sum += v;
    }
  }

  /* normalize so the filter preserves the overall brightness */
  if (sum > 0.0f)
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
	k[i][j] /= sum;

  return k;
}

float **sharpker(void)
{
  float **k = NULL;
  zeros(k, 3, 3);

  k[0][0] =  0; k[0][1] = -1; k[0][2] =  0;
  k[1][0] = -1; k[1][1] =  5; k[1][2] = -1;
  k[2][0] =  0; k[2][1] = -1; k[2][2] =  0;

  return k;
}

float **averageker(size_t n)
{
  if (n == 0) return NULL;

  float **k = NULL;
  zeros(k, n, n);

  float val = 1.0 / (float)(n*n);
  for (size_t i=0; i<n; ++i) for (size_t j=0; j<n; ++j) k[i][j] = val;

  return k;
}

float **sobelxker(void)
{
  float **k = NULL;
  zeros(k, 3, 3);

  k[0][0] = -1; k[0][1] = 0; k[0][2] = 1;
  k[1][0] = -2; k[1][1] = 0; k[1][2] = 2;
  k[2][0] = -1; k[2][1] = 0; k[2][2] = 1;

  return k;
}

float **sobelyker(void)
{
  float **k = NULL;
  zeros(k, 3, 3);

  k[0][0] = -1; k[0][1] = -2; k[0][2] = -1;
  k[1][0] =  0; k[1][1] =  0; k[1][2] =  0;
  k[2][0] =  1; k[2][1] =  2; k[2][2] =  1;

  return k;
}

float **sobelmag(float **img)
{
  if (!img) return NULL;

  float **mag = NULL;
  
  float **Kx = sobelxker();
  float **Ky = sobelyker();

  float **dx = conv(img, Kx);
  float **dy = conv(img, Ky);

  freemat(Kx);
  freemat(Ky);

  if (!dx || !dy) goto clean;

  const size_t r = matrows(img), c = matcols(img);
  createmat(mag, r, c);

  for (size_t i = 0; i<r; ++i)
    for (size_t j = 0; j<c; ++j)
      mag[i][j] = sqrtf(dx[i][j]*dx[i][j] + dy[i][j]*dy[i][j]);

 clean:
  freemat(dx);
  freemat(dy);
  return mag;
}

float **sobelori(float **img)
{
  if (!img) return NULL;
  float **ori = NULL;
  
  float **Kx = sobelxker();
  float **Ky = sobelyker();

  float **dx = conv(img, Kx);
  float **dy = conv(img, Ky);

  freemat(Kx);
  freemat(Ky);

  if (!dx || !dy) goto clean;

  const size_t r = matrows(img), c = matcols(img);
  createmat(ori, r, c);

  for (size_t i = 0; i<r; ++i)
    for (size_t j = 0; j<c; ++j)
      ori[i][j] = atan2f(dy[i][j], dx[i][j]);

 clean:
  freemat(dx);
  freemat(dy);
  return ori;
}

void rescaleu8(float **img)
{
  static const int L = 0;
  static const int H = 255;
  uint8_t lut[256];
  float M = matmax(img);
  float m = matmin(img);
  if (M == m) {
    for (int i = 0; i < 256; ++i) lut[i] = (uint8_t)L;
  } else {
    float scale = (float)(H - L) / (float)(M - m);
    for (int i = 0; i < 256; ++i) {
      float v = ((float)i - (float)m) * scale + (float)L;
      if (v < (float)L) v = (float)L;
      if (v > (float)H) v = (float)H;
      lut[i] = (uint8_t)(v + 0.5f);   /* round */
    }
  }
  for (size_t i=0; i<matrows(img); ++i)
    for (size_t j=0; j<matcols(img); ++j)
      img[i][j] = lut[(uint8_t)img[i][j]];
}

/* ===============================Convolution=============================== */

static void fft1d(double complex *a, size_t n, int inverse)
{
  if (n <= 1) return;

  /* bit-reversal permutation */
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double complex t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }

  /* butterflies */
  for (size_t len = 2; len <= n; len <<= 1) {
    double ang = 2.0 * M_PI / (double)len * (inverse ? +1.0 : -1.0);
    double complex wlen = cos(ang) + I * sin(ang);
    for (size_t i = 0; i < n; i += len) {
      double complex w = 1.0;
      for (size_t j = 0; j < len / 2; ++j) {
	double complex u = a[i + j];
	double complex v = a[i + j + len / 2] * w;
	a[i + j]             = u + v;
	a[i + j + len / 2]   = u - v;
	w *= wlen;
      }
    }
  }

  if (inverse) for (size_t i = 0; i < n; ++i) a[i] /= (double)n;
}

/* FFT 2D in-place  */
static void fft2d(double complex *a, size_t n, int inverse)
{
  for (size_t i = 0; i < n; ++i) fft1d(a + i * n, n, inverse); // rows
  
  double complex *col = xmalloc(n * sizeof(*col)); // cloumns
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) col[i] = a[i * n + j];
    fft1d(col, n, inverse);
    for (size_t i = 0; i < n; ++i) a[i * n + j] = col[i];
  }
  free(col);
}

static inline size_t nextpow2(size_t n)
{
  size_t p=1;
  while(p<n) p <<= 1;
  return p;
}

/* FFT based 2D convolution
 * conv_fft(m, k)[i,j] = sum_{u,v} m[i-u, j-v] * k[u, v] */
float **conv_fft(float **m, float **k)
{
  if (!m || !k) return NULL;

  const size_t R = matrows(m), C = matcols(m);
  const size_t P = matrows(k), Q = matcols(k);
  if (P == 0 || Q == 0) return NULL;
  if ((P & 1) == 0 || (Q & 1) == 0) return NULL;   

  const size_t fullR = R + P - 1;
  const size_t fullC = C + Q - 1;

  const size_t N = nextpow2(fullR > fullC ? fullR : fullC);
  if (N == 0) return NULL;

  double complex *A = xcalloc((size_t)N * N, sizeof(*A));
  double complex *B = xcalloc((size_t)N * N, sizeof(*B));

  /* copy m and k in A and B */
  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      A[i * N + j] = m[i][j];

  for (size_t i = 0; i < P; ++i)
    for (size_t j = 0; j < Q; ++j)
      B[i * N + j] = k[i][j];

  fft2d(A, N, 0);
  fft2d(B, N, 0);

  for (size_t i = 0; i < N * N; ++i) A[i] *= B[i];

  fft2d(A, N, 1);

  const size_t p = P / 2;
  const size_t q = Q / 2;

  float **cv = NULL;
  zeros(cv, R, C);

  for (size_t i = 0; i < R; ++i)
    for (size_t j = 0; j < C; ++j)
      cv[i][j] = (float)creal(A[(i + p) * N + (j + q)]);

  free(A);
  free(B);
  return cv;
}

/* Optimized convolution using lookup tables for small kernels.
 * It's not convenient to use this function with a PxQ kernel where P*Q>49
 * due to the size of the lookup tables. In that case the lookup table mul
 * would not fit in the cache, causing cache misses.
 * It returns float** or NULL on failure. Kernel sizes must be odd.
 * Actually the overhead is more than the gain starting to PxQ=21, so it's
 * more convenient to use an hardcoded conv that only works for 3x3 kernels */
float **conv_lut(float **m, float **k)
{
  const size_t r = matrows(m), c = matcols(m);
  const size_t P = matrows(k), Q = matcols(k);
  const size_t K = P * Q;
  const size_t p = P / 2, q = Q / 2;

  float **cv = NULL;
  zeros(cv, r, c);
  if (r == 0 || c == 0) return cv;

  float **mul = NULL;
  createmat(mul,256,K);
  for (int i = 0; i<256; ++i)
    for (size_t j = 0; j<K; ++j)
      mul[i][j] = (float)i * (*k)[j];   

  /* Flipped offsets */
  ssize_t *lut = xmalloc(K * sizeof(*lut));
  for (size_t x = 0; x < P; ++x)
    for (size_t y = 0; y < Q; ++y)
      lut[x*Q + y] = ((ssize_t)(P-1-x) - (ssize_t)p) * (ssize_t)c
	+ (ssize_t)(Q-1-y) - (ssize_t)q;


  if (r >= 2*p + 1 && c >= 2*q + 1) {
    for (size_t i = p; i<r-p; ++i) {
      size_t row = i * c;
      for (size_t j = q; j <c-q; ++j) {
	size_t a = row + j;
	float t = 0;
	for (size_t b = 0; b<K; ++b) t += mul[(int)(*m)[a+lut[b]]][b];
	(*cv)[a] = t;
      }
    }
  }

  freemat(mul);
  free(lut);
  return cv;
}


float **conv_standard(float **m, float **k)
{
  const size_t r = matrows(m),   c = matcols(m);
  const size_t p = matrows(k)/2, q = matcols(k)/2;

  if (p==3 && q==3){
    float **cv;
    zeros(cv, r, c);

    //size_t lut[] = {-r-1,-r,-r+1,-1,0,1,r-1,r,r+1};
    size_t lut[] = {r+1,r,r-1,1,0,+1,-r+1,-r,-r-1};
    float mul[256][9];

    for (unsigned short i = 0; i < 256; ++i)
      for (unsigned short j = 0; j < 9; ++j)
	mul[i][j] = i * (*k)[j];
        
    for (size_t a = 0; a<r*c; ++a){
      float t = 0;
      for (size_t b = 0; b<9; ++b) t += mul[(int)(*m)[lut[b]+a]][b];
      (*cv)[a] += t;
    }
    return cv;
  }  
  
  float **cv;
  zeros(cv, r, c);

  for (size_t i = 0; i < r; ++i){
    for (size_t j = 0; j < c; ++j) {
      float t = 0.0f;
      for (size_t x = 0; x < matrows(k); ++x) {
	ssize_t ii = (ssize_t)i + (ssize_t)x - (ssize_t)p;
	if (ii < 0 || ii >= (ssize_t)r) continue;
	for (size_t y = 0; y < matcols(k); ++y) {
	  ssize_t jj = (ssize_t)j + (ssize_t)y - (ssize_t)q;
	  if (jj < 0 || jj >= (ssize_t)c) continue;
	  t += m[ii][jj] * k[matrows(k)-1-x][matcols(k)-1-y];
	}
      }
      cv[i][j] = t;
    }
  }
  return cv;
}

float **conv(float **m, float **k)
{
    if (!m || !k) return NULL;

    const size_t R = matrows(m), C = matcols(m);
    const size_t P = matrows(k), Q = matcols(k);
    if (P == 0 || Q == 0) return NULL;

    if ((P & 1) == 0 || (Q & 1) == 0) return NULL; // even kernel

    const size_t op = R * C * P * Q; /* standard conv cost extimation */

    return op < FFT_THRESHOLD_CONV ? conv_standard(m, k) : conv_fft(m, k); 
}

/* ========================================================================= */

int **ind2rgb(indexed_t *img)
{
  if (!img || !img->i || !img->palette) return NULL;

  const size_t n = img->n;
  if (n == 0) return NULL;

  const size_t r = matrows(img->i), c = matcols(img->i);

  int *lut = xmalloc(n * sizeof(*lut));
  for (size_t k = 0; k < n; ++k) {
    int rr = (int)lrintf(fminf(255, fmaxf(0, img->palette[k][0])));
    int gg = (int)lrintf(fminf(255, fmaxf(0, img->palette[k][1])));
    int bb = (int)lrintf(fminf(255, fmaxf(0, img->palette[k][2])));
    lut[k] = (rr << 16) | (gg << 8) | bb;
  }

  int **out = NULL;
  createmat(out, r, c);

  for (size_t i = 0; i < r; ++i) {
    for (size_t j = 0; j < c; ++j) {
      uint16_t k = img->i[i][j];
      if ((size_t)k >= n) goto panic;
      out[i][j] = lut[k];
    }
  }

  free(lut);
  return out;

 panic:
  if (lut) free(lut);
  freemat(out);
  return NULL;
}

typedef struct {
  size_t  *pixels;
  size_t   count;    /* number of pixels in the parallelepiped */
  /* bounds */
  uint8_t  rmin, rmax;
  uint8_t  gmin, gmax;
  uint8_t  bmin, bmax;
} parallelepiped_t;

#define bounds(p, src)						\
  do {								\
    (p)->rmin = (p)->gmin = (p)->bmin = 255;			\
    (p)->rmax = (p)->gmax = (p)->bmax = 0;			\
    for (size_t _i = 0; _i < (p)->count; ++_i) {		\
      int _px = (src)[(p)->pixels[_i]];				\
      uint8_t _r = (uint8_t)((_px >> 16) & 0xFF);		\
      uint8_t _g = (uint8_t)((_px >>  8) & 0xFF);		\
      uint8_t _b = (uint8_t)( _px        & 0xFF);		\
      if (_r < (p)->rmin) (p)->rmin = _r;			\
      if (_r > (p)->rmax) (p)->rmax = _r;			\
      if (_g < (p)->gmin) (p)->gmin = _g;			\
      if (_g > (p)->gmax) (p)->gmax = _g;			\
      if (_b < (p)->bmin) (p)->bmin = _b;			\
      if (_b > (p)->bmax) (p)->bmax = _b;			\
    }								\
  } while (0)

static int cmpkey(const void *a, const void *b)
{
  size_t ka = *(const size_t *)a;
  size_t kb = *(const size_t *)b;
  return (ka > kb) - (ka < kb);
}

/* Shrink p in place along its longest axis, splitting at the median pixel.
 * The lower half stays in p, the upper half is written into u.
 * Returns 1 on success, 0 otherwise. */
static int shrink(parallelepiped_t *p, parallelepiped_t *u, const int *src)
{
  if (p->count < 2) return 0;

  /* Returns the index of the channel where parallelepiped is longer
   * 0 = red, 1 = green, 2 = blue */
  #define longest(p)						\
    ({								\
      int _dr = (p)->rmax - (p)->rmin;				\
      int _dg = (p)->gmax - (p)->gmin;				\
      int _db = (p)->bmax - (p)->bmin;				\
      int _ax;							\
      if (_dr >= _dg && _dr >= _db) _ax = 0;			\
      else if (_dg >= _db)          _ax = 1;			\
      else                          _ax = 2;			\
      _ax;							\
    })

  const int axis = longest(p);

  #undef longest


  const int shift = 16 - 8 * axis;          /* 16 = R, 8 = G, 0 = B */

  size_t *keys = xmalloc(p->count * sizeof(*keys));
  for (size_t i = 0; i < p->count; ++i) {
    size_t idx = p->pixels[i];
    uint64_t c = (uint64_t)((src[idx] >> shift) & 0xFF);
    /* puts c in the high part of the word and the original
     * index in the low part. */
    keys[i] = (c << 48) | (uint64_t)idx; 
  }

  qsort(keys,p->count,sizeof(*keys),cmpkey);

  /* Original pixel extraction, low part of the word */
  for(size_t i=0; i<p->count; ++i)
    p->pixels[i] = (size_t)(keys[i] & ((1ULL<<48)-1));

  free(keys);

  const size_t half = p->count / 2;

  u->pixels = p->pixels + half;
  u->count  = p->count - half;
  p->count  = half;

  bounds(p, src);
  bounds(u, src);
  return 1;
}


/* Euclidean squared distance */
#define colordist2(r1,g1,b1,r2,g2,b2)			\
  ((r1-r2)*(r1-r2) + (g1-g2)*(g1-g2) + (b1-b2)*(b1-b2))


static void average(const parallelepiped_t *p, const int *src, float out[3])
{
  if (p->count == 0) { out[0]=out[1]=out[2]=0.0f; return; }
  size_t sr = 0, sg = 0, sb = 0;
  for (size_t i = 0; i < p->count; ++i) {
    int px = src[p->pixels[i]];
    sr += (px >> 16) & 0xFF;
    sg += (px >>  8) & 0xFF;
    sb +=  px        & 0xFF;
  }
  const float w = 1.0 / (float)p->count;
  out[0] = (float)sr * w;
  out[1] = (float)sg * w;
  out[2] = (float)sb * w;
}

indexed_t *rgb2ind(int **img, size_t n)
{
  if (!img || n == 0) return NULL;

  const size_t R = matrows(img);
  const size_t C = matcols(img);
  const size_t N = R * C;
  if (N == 0) return NULL;

  /* The image is represented as one contiguous array; the parallelepiped
   * indices refer into it via a row-major mapping so the algorithm doesn't
   * depend on the layout of the matrix. */
  int *src = NULL;
  createvec(src, N);
  size_t k = 0;
  for(size_t i=0; i<R; ++i) for(size_t j=0; j<C; ++j) src[k++] = img[i][j];

  if (n > N) n = N;

  indexed_t *out = xmalloc(sizeof(*out));
  out->n = n;
  out->i = NULL;
  out->palette = NULL;

  size_t *indices = xmalloc(N * sizeof(*indices));
  for (size_t i=0; i<N; ++i) indices[i] = i;

  parallelepiped_t *pp = xcalloc(n, sizeof(*pp));
  pp[0].pixels = indices;
  pp[0].count = N;
  bounds(&pp[0], src);

  size_t npp = 1;

  /* chose the parallelepiped to shrink */
  while (npp < n) {
    ssize_t best = -1;
    int best_span = -1;

    for (size_t j=0; j<npp; ++j) {
      if (pp[j].count < 2) continue;

      int dr = pp[j].rmax - pp[j].rmin;
      int dg = pp[j].gmax - pp[j].gmin;
      int db = pp[j].bmax - pp[j].bmin;
      int span = dr > dg ? dr : dg;
      if (db > span) span = db;

      if (span > best_span) { best_span = span; best = (ssize_t)j; }
    }

    if (best < 0) break;

    parallelepiped_t upper = {0};
    if (!shrink(&pp[best], &upper, src)) break;

    pp[npp++] = upper;
  }


  /* palette building */
  out->palette = xmalloc(n * sizeof(*out->palette));
  for (size_t j = 0; j < npp; ++j)
    COLOR_CHOSE_FUNC(&pp[j], src, out->palette[j]);

  for (size_t j = npp; j < n; ++j) {
    out->palette[j][0] = out->palette[npp-1][0];
    out->palette[j][1] = out->palette[npp-1][1];
    out->palette[j][2] = out->palette[npp-1][2];
  }

  uint16_t *lut = xmalloc(N * sizeof(*lut));
  for (size_t j = 0; j < npp; ++j)
    for (size_t i = 0; i < pp[j].count; ++i)
      lut[pp[j].pixels[i]] = (uint16_t)j;

  uint16_t **iimg = NULL;
  createmat(iimg, R, C);
  for(size_t i=0; i<R; ++i) for(size_t j=0; j<C; ++j) iimg[i][j]=lut[i*C+j];

  out->i = iimg;

  free(lut);
  free(pp);
  free(indices);
  freevec(src);

  return out;
}


indexed_t *dithering(int **img, size_t n)
{
  if (!img || n == 0) return NULL;
  const size_t p = matrows(img), q = matcols(img);

  indexed_t *base = rgb2ind(img, n);
  if (!base) return NULL;

  float *pal = NULL;
  createvec(pal, n * 3);
  for (size_t k = 0; k < n; ++k) {
    pal[3*k + 0] = fminf(255.0f, fmaxf(0.0f, base->palette[k][0]));
    pal[3*k + 1] = fminf(255.0f, fmaxf(0.0f, base->palette[k][1]));
    pal[3*k + 2] = fminf(255.0f, fmaxf(0.0f, base->palette[k][2]));
  }

  float **R = NULL, **G = NULL, **B = NULL;
  xchannels(img,R,G,B);

  indexed_t *out = xmalloc(sizeof(*out));
  out->n = n;
  out->palette = base->palette;
  out->i = NULL;
  base->palette = NULL;
  createmat(out->i, p, q);

  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < q; ++j) {
      float r = fminf(255.0f, fmaxf(0.0f, R[i][j]));
      float g = fminf(255.0f, fmaxf(0.0f, G[i][j]));
      float b = fminf(255.0f, fmaxf(0.0f, B[i][j]));

      size_t best    = 0;
      float  best_d2 = INFINITY;
      for (size_t k = 0; k < base->n; ++k) {
	float d2 = colordist2(r, g, b,
			      pal[3*k + 0], pal[3*k + 1], pal[3*k + 2]);
	if (d2 < best_d2) { best_d2 = d2; best = k; }
      }

      out->i[i][j] = (uint16_t)best;

      /* errors */
      float er = r - pal[3*best + 0];
      float eg = g - pal[3*best + 1];
      float eb = b - pal[3*best + 2];

      #define DIFFUSE(di, dj, w)                          \
        do {                                              \
          size_t _ni = (size_t)((ssize_t)i + (di));       \
          size_t _nj = (size_t)((ssize_t)j + (dj));       \
          if (_ni < p && _nj < q) {                       \
            R[_ni][_nj] += er * (w);                      \
            G[_ni][_nj] += eg * (w);                      \
            B[_ni][_nj] += eb * (w);                      \
          }                                               \
        } while (0)

      DIFFUSE(0, +1, 7.0f / 16.0f);
      DIFFUSE(+1, -1, 3.0f / 16.0f);
      DIFFUSE(+1,  0, 5.0f / 16.0f);
      DIFFUSE(+1, +1, 1.0f / 16.0f);

      #undef DIFFUSE
    }
  }

  freevec(pal);
  freemat(R); freemat(G); freemat(B);
  return out;
}


#endif // IMG_IMPLEMENTATION
#endif // IMG_H
