# [Re] Curve-fitting with piecewise parametric cubics

This repository provides a reproduction of the classic paper [Curve-fitting with piecewise parametric cubics](https://dl.acm.org/doi/10.1145/800059.801153) by Michael Plass and Maureen Stone (1983). It implements a method based on dynamic programming for automatically segmenting and fitting piecewise cubic Bézier curves to a sequence of 2D data points.  

The code supports visualization of the curve-fitting process, uses the Ramer–Douglas–Peucker algorithm to simplify point sets, and exports both the fitted curves and an annotated plot.

## Features

- Sliding window tangent estimation  
- Dynamic programming knot selection  
- Bézier curve fitting with optional RDP simplification  
- Saves curve parameters to file  
- Supports alternative test figures

## Getting Started

You will need **Python 3.10+** and these packages:

```bash
pip install numpy matplotlib tqdm
```

## Usage
You can run the code like this:

```bash
python draw_single_image.py --input_file examples/S-trace.txt --output_file outputs_svg/S_svg.png
```

## Testing on Other Figures
You can experiment with other point data stored in the examples folder. For instance, to test on C.txt, run:

```bash
python draw_single_image.py --input_file examples/C.txt --output_file outputs_svg/C_txt.png
```

The output curves shall be in outputs.

## Acknowledgment
The code was written by Daniel Perazzo and Davi Guimarães Nunes Sena Castro.

This project is the product of a [course](https://lhf.impa.br/cursos/tmg/) at held IMPA in 2025.
