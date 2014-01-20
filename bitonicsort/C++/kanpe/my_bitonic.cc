#include <iostream>

void input(void);
void show(void);
void sort(void);
void sort_step(int step);
void sort_zone(int start, int end, int step, bool is_red_zone);
void red_sort(int left_idx, int right_idx);
void blue_sort(int left_idx, int right_idx);

const int MAX_N = 1<<20;
const int DUMMY = 1<<20;
int N;
int max;
int array[MAX_N];

int main(void) {
    input();
    sort();
    show();
    return 0;
}

void sort(void) {
    for (int i = 2; i <= max; i *= 2) {
        sort_step(i); 
    }
}

void sort_step(int step) {
    printf("----------------[%d]\n", step);
    bool flags[max];
    bool red_flag = true;
    for (int i = 0; i < max; i += step) {
        for (int k = i; k < i+step; k++) {
            flags[k] = red_flag; 
        }
        red_flag = !red_flag;
    }
    for (int i = 0; 1 < step; step /= 2, i++) {
        printf("------[%d]\n", step);
        for (int j = 0; j < max; j += step) {
            int start = j;
            int end   = start + step - 1;
            sort_zone(start, end, step, flags[j]);
        }
    }
}

// red_zone  : 昇順 small -> big
// !red_zone : 降順 big   -> small
void sort_zone(int start, int end, int step, bool is_red_zone) {
    int cnt = (end - start + 1) / 2;
    for (int i = 0; i < cnt; i++) {
        printf("[%02d] and [%02d] comparison ", start+i, start+i+(step/2)); 
        is_red_zone ? printf(" RED!!\n") : printf("BLUE!!\n");
        int left_idx  = start + i;
        int right_idx = start + i + (step/2);
        (is_red_zone) ? red_sort(left_idx, right_idx) : blue_sort(left_idx, right_idx);
    }
}

// small -> big
void red_sort(int left_idx, int right_idx) {
    int l = array[left_idx];
    int r = array[right_idx];
    if (l <= r) {
        return;
    }
    array[left_idx]  = r;
    array[right_idx] = l;
}

// big -> small
void blue_sort(int left_idx, int right_idx) {
    int l = array[left_idx];
    int r = array[right_idx];
    if (r <= l) {
        return;
    }
    array[left_idx]  = r;
    array[right_idx] = l;
}

void show(void) {
    //printf("16個ずつ表記\n");
    printf("[");
    for (int i = 0; i < max; i++) {
        if (i%16 == 0) {
            printf("\n"); 
        }
        printf("%02d, ", array[i]); 
    }
    printf("\n]\n");
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
