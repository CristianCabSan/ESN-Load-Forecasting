# Simple Echo State Network implementations

These minimalist self-contained source codes in different programming
languages demonstrate the simplicity and power of implementing and
applying [Echo State
Networks](http://www.scholarpedia.org/article/Echo_state_network)
"from scratch". This is arguably the simplest form of recurrent neural
network learning. The source codes are intended for education and
instruction, but can also be easily adapted for practical purposes.

As a demo, they learn to predict
[Mackey-Glass](http://www.scholarpedia.org/article/Mackey-Glass_equation)
chaotic time series (delay=17) with a remarkable accuracy. The data
needed for the code are available here:
[MackeyGlass_t17.txt](MackeyGlass_t17.txt) (or
[.zip](MackeyGlass_t17.txt.zip), needs unpacking).

The source codes in plain:

## Julia: [minimalESN.jl](minimalESN.jl).

Requires [Julia](https://julialang.org/) (tested on versions 1.0.5 and 1.4.2) with [Plots](https://docs.juliaplots.org/latest/) plotting library (multiple output [backends](https://docs.juliaplots.org/latest/backends/) available). They are all free.

## Matlab or Octave: [minimalESN.m](minimalESN.m).
Requires [Matlab](https://www.mathworks.com/products/matlab/) (tested on versions R2008b, R2014b, and R2016a) or [Octave](https://www.gnu.org/software/octave/) (tested on version 4.0.0). Octave is free.

## Scientific Python: [minimalESN.py](minimalESN.py).
Requires [Python](https://www.python.org/) (tested on versions 2.7.17, 3.5.4, 3.7.4, and 3.8.3) with general [scientific](https://www.scipy.org/) libraries: [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/) (optional), and [Matplotlib](https://matplotlib.org/). They are all free.

## R programming language: [minimalESN.r](minimalESN.r).
Requires [R programming environment](https://www.r-project.org/) (tested on versions 2.15.1 and 4.0.2). It’s free.

The program codes above are distributed under a friendly [MIT license](LICENSE).