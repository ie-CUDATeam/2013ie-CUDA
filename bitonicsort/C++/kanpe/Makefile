#CFLAGS = -W -Wno-sign-compare
#CFLAGS = -W
CFLAGS = -W -O2
#CFLAGS = -g
#CC = gcc
CC = g++

all:
	$(CC) $(CFLAGS) -o a.out bitonic.cc

test:
	./a.out < input

test2:
	./a.out < input2

clean:
	rm ./a.out

