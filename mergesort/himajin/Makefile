#CFLAGS = -W -Wno-sign-compare
#CFLAGS = -W
CFLAGS = -W -O2
#CFLAGS = -g
#CC = gcc
CC = g++

all:
	$(CC) $(CFLAGS) -o merge merge.cc
	nvcc -o cuda_merge -arch=sm_20 merge.cu

test:
	./a.out < input0

test1:
	./a.out < input1

test2:
	./a.out < input2

test3:
	./a.out < input3

clean:
	rm ./a.out

