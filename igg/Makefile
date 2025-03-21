Y= freetype
S=-DSTANDALONE

CC= cc
CFLAGS= -O2 -std=c99 -pedantic -Wall -Wextra -I $Y/include $S -I -llua5.4 -I /usr/include/freetype2
LIBS= -lfreetype -lz -lbz2 -llua5.4 -lm

P= fd
L= ps.lua
O= out
G= g
F= /System/Library/Fonts/NewYorkItalic.ttf
F= /System/Library/Fonts/NewYork.ttf
F= /System/Library/Fonts/Supplemental/Times New Roman Italic.ttf
F= /System/Library/Fonts/Supplemental/Times New Roman.ttf
F= /System/Library/Fonts/Times.ttc
F= /usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf

all:	run pdf

test:
	$(CC) $(CFLAGS) main.c $(LIBS)
	./a.out '$G' '$F'

r:	$P
	./$P $L '$G' '$F'

run:	$P
	./$P $L '$G' '$F' > $O.eps

pdf:
	epstopdf $O.eps
	open $O.pdf

$P:	$P.c outline.c outline.h
	$(CC) $(CFLAGS) -o $P $P.c $(LIBS)

clean:
	-rm -f $P $P.o $O.txt $O.ps $O.eps $O.pdf a.out *.o *.so

so:
	$(CC) $(CFLAGS) -o outline.so -bundle -undefined dynamic_lookup $P.c $(LIBS)
