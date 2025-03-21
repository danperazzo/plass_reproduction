/*
* inspect.c
* inspect glyph geometry using libschrift internals
* https://github.com/lhf/libschrift-show
* This code is hereby placed in the public domain and also under the MIT license.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "schrift.c"

static const char *progname;

static void fatal(const char *message)
{
	fprintf(stderr,"%s: %s\n",progname,message);
	exit(1);
}

static void dump(Outline *outl, unsigned int n, int bbox[4])
{
	int M=10;
	unsigned int i;
	printf("%%!PS-Adobe-2.0 EPSF-2.0\n");
	printf("%%%%BoundingBox: %d %d %d %d\n",bbox[0]-M,bbox[1]-M,bbox[2]+M,bbox[3]+M);
	printf("/p { 2 0 360 arc fill } bind def\n");
	printf("/q { 5 0 360 arc fill } bind def\n");
	printf("/l { moveto lineto stroke } bind def\n");
	printf("\n");
	printf("%% data %d lines %d curves\n",n,outl->numCurves);;
	for (i = 0; i < n; ++i) {
		Line line=outl->lines[i];
		Point a = outl->points[line.beg];
		Point b = outl->points[line.end];
		printf("%% L %g %g %g %g\n",a.x,a.y,b.x,b.y);
	}
	for (i = 0; i < outl->numCurves; ++i) {
		Curve curve=outl->curves[i];
		Point a = outl->points[curve.beg];
		Point b = outl->points[curve.ctrl];
		Point c = outl->points[curve.end];
		printf("%% Q %g %g %g %g %g %g\n",a.x,a.y,b.x,b.y,c.x,c.y);
	}
	printf("\n");
	printf("%% %d segments\n",outl->numLines);
	for (i = 0; i < outl->numLines; ++i) {
		Line line=outl->lines[i];
		Point a = outl->points[line.beg];
		Point b = outl->points[line.end];
		printf("%g %g %g %g l\n",a.x,a.y,b.x,b.y);
	}
	printf("\n");
	printf("%% points on segments\n");
	printf("1 1 1 setrgbcolor\n");
	for (i = 0; i < outl->numLines; ++i) {
		Line line=outl->lines[i];
		Point a = outl->points[line.beg];
		//Point b = outl->points[line.end];
		printf("%g %g p\n",a.x,a.y);
		//printf("%g %g p\n",b.x,b.y);
	}
	printf("\n");
	printf("%% points on curve\n");
	printf("0 0 0 setrgbcolor\n");
	for (i = 0; i < n; ++i) {
		Line line=outl->lines[i];
		Point a = outl->points[line.beg];
		//Point b = outl->points[line.end];
		printf("%g %g q\n",a.x,a.y);
		//printf("%g %g q\n",b.x,b.y);
	}
	for (i = 0; i < outl->numCurves; ++i) {
		Curve curve=outl->curves[i];
		Point a = outl->points[curve.beg];
		//Point c = outl->points[curve.end];
		printf("%g %g q\n",a.x,a.y);
		//printf("%g %g q\n",c.x,c.y);
	}
#if 0
	printf("\n");
	printf("%% control points\n");
	printf("0.5 setgray\n");
	for (i = 0; i < outl->numCurves; ++i) {
		Curve curve=outl->curves[i];
		Point a = outl->points[curve.beg];
		Point b = outl->points[curve.ctrl];
		Point c = outl->points[curve.end];
		printf("%g %g %g %g l\n",a.x,a.y,b.x,b.y);
		printf("%g %g %g %g l\n",c.x,c.y,b.x,b.y);
		printf("%g %g q\n",b.x,b.y);
	}
#endif
	printf("\n");
	printf("showpage\n");
	printf("%%%%EOF\n");
}

static int sft_inspect(const SFT *sft, SFT_Glyph glyph)
{
	uint_fast32_t outline;
	Outline outl;
	int bbox[4];
	if (outline_offset(sft->font, glyph, &outline) < 0)
		return -1;
	if (!outline)
		return 0;
	if (glyph_bbox(sft, outline, bbox) < 0)
		return -1;
	memset(&outl, 0, sizeof outl);
	if (init_outline(&outl) < 0)
		goto failure;
	if (decode_outline(sft->font, outline, 0, &outl) < 0)
		goto failure;
	int n=outl.numLines;
	if (tesselate_curves(&outl) < 0)
		goto failure;
	dump(&outl,n,bbox);
	free_outline(&outl);
	return 0;
failure:
	free_outline(&outl);
	return -1;
}

static void inspect(SFT *sft, const char *filename, double size, SFT_UChar codepoint)
{
	SFT_Font *font = sft_loadfile(filename);
	if (font == NULL) fatal("sft_loadfile failed");
	size = font->unitsPerEm;
	sft->font = font;
	sft->xScale = size;
	sft->yScale = size;
	sft->xOffset = 0.0;
	sft->yOffset = 0.0;
	sft->flags = 0;
	SFT_Glyph glyph;
	if (sft_lookup(sft,codepoint,&glyph) < 0) fatal("sft_lookup failed");
	if (sft_inspect(sft,glyph) < 0) fatal("sft_inspect failed");
	sft_freefont(sft->font);
}

int main(int argc, char *argv[])
{
	progname = argv[0];
	if (argc!=4) fatal("fontfile size_in_px glyph");
	const char *fontfile = argv[1];
	double size = atof(argv[2]);
	const char *s = argv[3];
	SFT_UChar codepoint = (s[0]=='#' && s[1]!=0) ? atof(argv[3]+1) : s[0];
	SFT sft;
	inspect(&sft,fontfile,size,codepoint);
	return 0;
}
