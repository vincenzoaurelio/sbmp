# sbmp
sbmp is a free simple image manipulation program for the command line.
## Compilation
- sbmp only depends on [SDL2](https://www.libsdl.org), which must be installed
  before compilation.
- To compile sbmp run `make`.
- It's possible to compile sbmp without any grphic interface with `make nogui`

## img lib
[`img.h`](./img.h), used by sbmp, is an [stb-style](https://github.com/nothings/stb/blob/master/README.md)
header-only library. 
## Usage
```text
./sbmp INPUT [-o OUTPUT] [options...]
  -o FILE    output image
  -h         help

Unary operations:
  -g         grayscale
  -G N[:S]   conv with gaussian kernel N with optional sigma S
  -A N       conv with average kernel N
  -S         conv with sharpening kernel
  -x         conv with Sobel X kernel
  -y         conv with Sobel Y kernel
  -m         Sobel magnitude (on grayscale)
  -O         Sobel orientation (on grayscale, visualized as HSV)
  -p K       multiply by scalar K (saturating)
  -a K       add scalar K (saturating)
  -s         square
  -r         square root
  -R         rescale to [0,255]

Binary operations:
  -B IMG     min
  -T IMG     max
  -D IMG     saturated sum
  -E IMG     average (normalized sum)
  -F IMG     saturated subtraction
  -H IMG     absolute difference
  -J IMG     multiply (normalized)
```

## Examples
- Convert an image to grayscale and write output on disk
  ```bash
  ./sbmp ./imgs/lena.ppm -g -o ./imgs/lena-gray.ppm
  ```
- Apply a Gaussian blur
  ```bash
  ./sbmp ./imgs/baboon.ppm -G 5
  ```
- Sharpen an image and write output on disk
  ```bash
  ./sbmp ./imgs/peppers.ppm -S -o ./imgs/peppers-sharpened.ppm
  ```
- Calculate the Sobel magnitude on a grayscale image and write output on disk
  ```bash
  ./sbmp ./imgs/peppers.ppm -g -m -o ./imgs/peppers-mag.ppm
  ```
- Multiply pixel values by a scalar
  ```bash
  ./sbmp ./imgs/peppers2.ppm -p 2
  ```
## License
This project is licensed under the BSD 3-Clause License - see the
[LICENSE](LICENSE) file for details.