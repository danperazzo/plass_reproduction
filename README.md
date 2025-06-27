# Bezier Curve Fitting - Piecewise Parametric Cubics Reproduction

This repository provides a reproduction of the classic algorithm described in **“Piecewise Parametric Cubics for Data Fitting”** by Michael Plass and Maureen Stone (1983). It implements a dynamic-programming-based method for automatically segmenting and fitting cubic Bézier curves to a set of 2D data points.  

The script supports visualization of the curve-fitting process, uses the Ramer–Douglas–Peucker (RDP) algorithm to simplify point sets, and exports both the fitted curves and an annotated plot.

---

## Features

✅ Sliding window tangent estimation  
✅ Dynamic programming knot selection  
✅ Bézier curve fitting with optional RDP simplification  
✅ Saves curve parameters to file  
✅ Supports alternative test figures

---

## Getting Started

You will need **Python 3.10+** and these packages:

```bash
pip install numpy matplotlib
```
Usage
The provided script is named example.py. You can run it like this:

```bash
python draw_single_image.py --input_file examples/S-trace.txt --output_file outputs_svg/S_svg.png
```

Testing on Other Figures
You can experiment with other point data stored in the examples folder. For instance, to test on S.txt, run:

```bash
python draw_single_image.py --input_file examples/C.txt --output_file outputs_svg/C_txt.png
```

The output curves shall be in outputs.

Acknowledgment
This project is a reproduction of the algorithm presented in:

Michael Plass and Maureen Stone. “Piecewise Parametric Cubics for Data Fitting.” SIGGRAPH, 1983.
