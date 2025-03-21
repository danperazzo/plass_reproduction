/*
* show.c
* show text using libschrift
* https://github.com/lhf/libschrift-show
* This code is hereby placed in the public domain and also under the MIT license.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "schrift.h"

#define max(a,b) ((a) > (b) ? a : b)

static const char *progname;

static void fatal(const char *message)
{
	fprintf(stderr,"%s: %s\n",progname,message);
	exit(1);
}

static void loadfont(SFT *sft, const char *filename, double size, SFT_LMetrics *lmtx)
{
	SFT_Font *font = sft_loadfile(filename);
	if (font == NULL) fatal("sft_loadfile failed");
	sft->font = font;
	sft->xScale = size;
	sft->yScale = size;
	sft->xOffset = 0.0;
	sft->yOffset = 0.0;
	sft->flags = SFT_DOWNWARD_Y;
	if (sft_lmetrics(sft,lmtx) < 0) fatal("sft_lmetrics failed");
}

static void loadglyph(const SFT *sft, SFT_UChar codepoint, SFT_Glyph *glyph, SFT_GMetrics *metrics)
{
	if (sft_lookup(sft,codepoint,glyph) < 0) fatal("sft_lookup failed");
	if (sft_gmetrics(sft,*glyph,metrics)< 0) fatal("sft_gmetrics failed");
}

static void newimage(SFT_Image *image, int width, int height, int color)
{
	size_t size = (size_t) (width*height);
	void *pixels  = malloc(size);
	image->pixels = pixels;
	image->width  = width;
	image->height = height;
	memset(pixels,color,size);
}

static void saveimage(SFT_Image *image, const char *filename)
{
	size_t size = (size_t) (image->width*image->height);
	FILE *f = fopen(filename,"wb");
	if (f == NULL) fatal("fopen outfile failed");
	fprintf(f,"P5\n%d %d\n255\n",image->width,image->height);
	fwrite(image->pixels,size,1,f);
	fclose(f);
}

static void copyimage(SFT_Image *dest, const SFT_Image *source, int x0, int y0, int color)
{
	unsigned char *d=dest->pixels;
	unsigned char *s=source->pixels;
	d+=x0+y0*dest->width;
	int y;
	for (y=0; y<source->height; y++)
	{
		int x;
		for (x=0; x<source->width; x++)
		{
			double t=s[x]/255.0;
			d[x]=(1.0-t)*d[x]+t*color;
		}
		d+=dest->width;
		s+=source->width;
	}
}

int main(int argc, char *argv[])
{
	int fg=0,bg=255;
	const char *fontfile;
	double size;
	const char *message;
	const char *outfile;
	SFT sft;
	SFT_Image canvas;
	SFT_LMetrics lmtx;

	progname = argv[0];
	if (argc!=5) fatal("fontfile size_in_px message outfile");
	fontfile = argv[1];
	size = atof(argv[2]);
	message = argv[3];
	outfile = argv[4];

	loadfont(&sft,fontfile,size,&lmtx);

	size_t k;
	size_t n=strlen(message);
	int margin = 100;
	int width  = 0;
	int height = lmtx.ascender + lmtx.lineGap;
	int lwidth = 0;
	for (k=0; k<n; k++)
	{
		if (message[k]=='\\' && message[k+1]=='n')
		{
			k++;
			width = max(width,lwidth);
			height += lmtx.ascender - lmtx.descender + lmtx.lineGap;
			lwidth = 0;
			continue;
		}
		SFT_UChar cp = (SFT_UChar) message[k];
		SFT_Glyph gid;
		SFT_GMetrics mtx;
		loadglyph(&sft,cp,&gid,&mtx);
		if (lwidth==0 && mtx.leftSideBearing<0 && margin<-mtx.leftSideBearing) margin-=mtx.leftSideBearing;
		lwidth += max(mtx.advanceWidth,mtx.minWidth);
	}
	height += -lmtx.descender + lmtx.lineGap;
	width = max(width,lwidth);
	width += 2*margin;
	height+= 2*margin;
	newimage(&canvas,width,height,bg);

	double x = margin;
	double y = margin + lmtx.ascender + lmtx.lineGap;
	SFT_Glyph ogid=0;
	for (k=0; k<n; k++)
	{
		if (message[k]=='\\' && message[k+1]=='n')
		{
			k++;
			x  = margin;
			y += lmtx.ascender - lmtx.descender + lmtx.lineGap;
			ogid=0;
			continue;
		}
		SFT_Image image;
		SFT_UChar cp = (SFT_UChar) message[k];
		SFT_Glyph gid;
		SFT_GMetrics mtx;
		SFT_Kerning kerning;
		loadglyph(&sft,cp,&gid,&mtx);
		newimage(&image,mtx.minWidth,mtx.minHeight,0);
		sft_render(&sft,gid,image);
		sft_kerning(&sft,ogid,gid,&kerning);
		x += kerning.xShift;
		copyimage(&canvas,&image,x+mtx.leftSideBearing,y+mtx.yOffset,fg);
		x += mtx.advanceWidth;
		free(image.pixels);
		ogid=gid;
	}
	saveimage(&canvas,outfile);

	free(canvas.pixels);
	sft_freefont(sft.font);
	return 0;
}
