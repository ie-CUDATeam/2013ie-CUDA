#include <iostream>
#include <time.h>
#include <algorithm>

const int MAX_N = 1<<20;
const int DUMMY = 1<<20;
int N;
int max;
int array[MAX_N];

void input(void);

int main(void) {
    clock_t start, stop;
    input();
    start = clock();
    std::sort(array, array+N);
    stop  = clock();
    double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

void input(void) {
    scanf("%d", &N);
    for (int i = 0; i < MAX_N; i++) {
        array[i] = DUMMY;
    }
    for (int i = 0; i < N; i++) {
        int a;
        scanf("%d", &a);
        array[i] = a;
    }
    max = 2;
    while (max < N) max *= 2;
}
