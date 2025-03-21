libschrift-show
===============

This is a program to test the features of [libschrift](https://github.com/tomolt/libschrift), a lightweight TrueType font rendering library. Unlike the demos in libschrift, show.c does not depend on any graphics environment: it generates a [PGM file](https://en.wikipedia.org/wiki/Netpbm#File_formats), which can be viewed virtually everywhere.

There is also inspect.c, a tool to inspect glyph geometry using libschrift internals. It also shows the polygonal approximation computed by libschrift for outline curves. The output is in PostScript but it can be easily changed to use other vector graphics formats. See also my [igg](https://github.com/lhf/igg).

The code in show.c and inspect.c is hereby placed in the public domain and also under the MIT license.

See the LICENSE file for copyright and license details on the code in schrift.c and schrift.h.
