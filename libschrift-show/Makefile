# show text using libschrift

# default args; paths for macOS
F= /System/Library/Fonts/NewYork.ttf
F= /System/Library/Fonts/NewYorkItalic.ttf
F= /System/Library/Fonts/Supplemental/Times New Roman Italic.ttf
S= 500
M= Hello from\nlibschrift!\nabcdefghij
O= out.pgm

CFLAGS = -O2 -std=c99 -pedantic -Wall -Wextra # -Wconversion

all:	run

run:	show
	./show '$F' $S '$M' $O
	open $O

show:	show.c schrift.o
	$(CC) $(CFLAGS) -o $@ $@.c schrift.o -lm

dump:	inspect
	./inspect '$F' $S '$M' > out.eps
	epstopdf out.eps
	open out.pdf

inspect: inspect.c schrift.c
	$(CC) $(CFLAGS) -o $@ $@.c -lm

schrift.o: schrift.c schrift.h

clean:
	rm -f show schrift.o out.pgm inspect out.eps out.pdf

.PHONY: all run clean

