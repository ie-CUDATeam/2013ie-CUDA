#include <iostream>
#include <iomanip>
#include <vector>

void input(void);
void show_vec(void);
void step(const int block);
//void ascending_order(int n);
//void descending_order(int n);

const int SHOW_DIGIT  = 2;
const int DUMMY_VALUE       = 99;
std::vector<int> vec;
int N;
int main(void) {
    input();
    //std::cout << "Before" << std::endl;
    //show_vec();
    for (int i = 2; i <= N; i*=2) {
        step(i);
        show_vec();
        std::cout << std::endl;
    }
    return 0;
}

void step(const int block) {
    //printf("-------------------------[%d]\n", block);
    for (int step = block/2; 1 <= step; step/=2) {
        bool blue_flag = true;
        //printf("----------[%d]\n", step);
        for (int idx = 0; idx < N; idx++) {
            int e = idx^step;
            if (idx < e) {
                int v1 = vec[idx];
                int v2 = vec[e];
                if ((idx&block) != 0) {
                    //printf("[%02d] and [%02d] comparison ", idx, e);
                    //printf("BLUE!!\n");
                    if (v1 < v2) {
                        vec[e] = v1; 
                        vec[idx] = v2; 
                    }
                } else {
                    //printf("[%02d] and [%02d] comparison ", idx, e);
                    //printf(" RED!!\n");
                     if (v2 < v1) {
                        vec[e] = v1; 
                        vec[idx] = v2; 
                    }
                }
            }
        }
    }
}

void show_vec(void) {
    std::cout << "[";
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(SHOW_DIGIT) << std::setfill('0') << vec[i];
        if (i < N-1) {
            std::cout << ", "; 
        }
    }
    std::cout << "]" << std::endl;
}

void input(void) {
    int temp;
    std::cin >> temp;
    N = 2;
    while (N < temp) {
        N *= 2;
    }
    std::cout << "orginal data size : " << temp << std::endl;
    std::cout << "dummy   data size : " << N-temp << std::endl;
    vec = std::vector<int>(N, DUMMY_VALUE);
    for (int i = 0; i < temp; i++) {
        int a;
        std::cin >> a;
        vec[i] = a;
    }
}
