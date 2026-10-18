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
 n * inside the header.
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
    size_t _r = matrows(dest), _c=matcols(dest);	\
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

/* =============================vector sorting============================== */
/* returns index of n if n is found in the sorted vector vec, -1 otherwise */
#define binarysearch(vec,n)				\
  ({							\
    ssize_t _index =-1;					\
    size_t _l = 0;					\
    size_t _r = vecsize(vec);				\
    while (_l < _r) {					\
      size_t _mid = _l + (_r-_l)/2;			\
      if ((vec)[_mid]==(n)) {_index = _mid; break;}	\
      if ((vec)[_mid]>(n)) _r = _mid;			\
      else _l=_mid+1;					\
    }							\
    _index;						\
  })

/* returns the greatest element, in absolute value, in a vector.
 * If the vector's length is 0, 0 is returned */
#define vmax(vec)				\
  ({						\
    double _max=0;				\
    for (size_t _i=0;_i<vecsize(vec);++_i)	\
      _max=fmax(_max, fabs((vec)[_i]));		\
    (typeof((vec)[0]))_max;			\
  })

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


float **mpow(float **mat, char exponent); // matrix power

/* -----------------------------------img----------------------------------- */

/* Read a binary PPM (P6) into an int** */
int **loadppm(const char *path);

/* Convert a packed-RGB int image to float grayscale using weights
 * Y = 0.299*R + 0.587*G + 0.114*B.
 * Returns a float** with one channel per pixel. */
float **rgb2grey(int **img);
int **pack_image(float **R, float **G, float **B);
void rescaleu8(float **I);

/* ------------------------------ convolution ------------------------------ */
#ifdef DEBUG
float **conv_standard(float **m, float **k);
float **conv_fft(float **m, float **k);
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
float **sobelmag(float **I);
/* Sobel gradient orientation.
 * sobelori(I) = atan(dy/dx)
 * returns a float** of the same shape as I, or NULL on failure. */
float **sobelori(float **I);

#define red(p) (float)((p>>16) & 0xFF)
#define green(p) (float)((p>>8) & 0xFF)
#define blu(p) (float)((p & 0xFF))

#define min(i,j) (((i) > (j)) ? (j) : (i))
#define max(i,j) (((i) < (j)) ? (j) : (i))

#define extractred(img)						\
  ({								\
    float **_R = NULL;						\
    if (img) {							\
      size_t _r = matrows(img), _c = matcols(img);		\
      createmat(_R, _r, _c);					\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          _R[_i][_j] = (float)(((img)[_i][_j] >> 16) & 0xFF);	\
    }								\
    _R;								\
  })

#define extractgreen(img)					\
  ({								\
    float **_G = NULL;						\
    if (img) {							\
      size_t _r = matrows(img), _c = matcols(img);		\
      createmat(_G, _r, _c);					\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          _G[_i][_j] = (float)(((img)[_i][_j] >> 8) & 0xFF);	\
    }								\
    _G;								\
  })

#define extractblu(img)					\
  ({							\
    float **_B = NULL;					\
    if (img) {						\
      size_t _r = matrows(img), _c = matcols(img);	\
      createmat(_B, _r, _c);				\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          _B[_i][_j] = (float)((img)[_i][_j] & 0xFF);	\
    }							\
    _B;							\
  })

/* saturation to 255 */
#define mulscalar(I,k)					\
  do {							\
    size_t _r = matrows(I), _c = matcols(I);		\
    if ((k) >= 0) {					\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min(255, (I)[_i][_j] * (k));	\
    }							\
  } while (0)

/* saturation to 255 */
#define sumscalar(I,k)					\
  do {							\
    size_t _r = matrows(I), _c = matcols(I);		\
    if ((k) >= 0) {					\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min(255, (I)[_i][_j] + (k));	\
    }							\
  } while (0)


#define imgmin(I,J)					\
  do {							\
    size_t _r = matrows(I), _c = matcols(I);		\
    if (matrows(J) == _r && matcols(J) == _c) {		\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = min((I)[_i][_j], (J)[_i][_j]);	\
    }							\
  } while (0)


#define imgmax(I,J)					\
  do {							\
    size_t _r = matrows(I), _c = matcols(I);		\
    if (matrows(J) == _r && matcols(J) == _c) {		\
      for (size_t _i = 0; _i < _r; ++_i)		\
        for (size_t _j = 0; _j < _c; ++_j)		\
          (I)[_i][_j] = max((I)[_i][_j], (J)[_i][_j]);	\
    }							\
  } while (0)

/* saturation to 255 */
#define satsum(I,J)						\
  do {								\
    size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = min(255, (I)[_i][_j] + (J)[_i][_j]);	\
    }								\
  } while (0)

/* normalized in [0, 255] */
#define normsum(I,J)						\
  do {								\
    size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = ((I)[_i][_j] + (J)[_i][_j]) / 2;	\
    }								\
  } while (0)

/* saturation to 255 */
#define satsubtraction(I,J)					\
  do {								\
    size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = max(0, (I)[_i][_j] - (J)[_i][_j]);	\
    }								\
  } while (0)


#define subtraction(I,J)					\
  do {								\
    size_t _r = matrows(I), _c = matcols(I);			\
    if (matrows(J) == _r && matcols(J) == _c) {			\
      for (size_t _i = 0; _i < _r; ++_i)			\
        for (size_t _j = 0; _j < _c; ++_j)			\
          (I)[_i][_j] = fabs((I)[_i][_j] - (J)[_i][_j]);	\
    }								\
  } while (0)

/* normalized in [0, 255]  */
#define mult(I,J)						\
  do {								\
    size_t _r = matrows(I), _c = matcols(I);			\
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
    size_t _r = matrows(I), _c = matcols(I);	\
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
  size_t m = matrows(mat);
  size_t n = matcols(mat);
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
  float **R = extractred(img);
  float **G = extractgreen(img);
  float **B = extractblu(img);


  size_t r = matrows(img), c = matcols(img);
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

  size_t r = matrows(R), c = matcols(R);
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

  size_t r = matrows(img);
  size_t c = matcols(img);
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

  size_t r = matrows(img);
  size_t c = matcols(img);
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

float **conv_standard(float **m, float **k)
{
  size_t r = matrows(m),   c = matcols(m);
  size_t p = matrows(k)/2, q = matcols(k)/2;

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
#endif // IMG_IMPLEMENTATION
#endif // IMG_H
