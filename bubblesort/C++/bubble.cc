#include <iostream>

// INPUT
const int SIZE  = 10;
int array[SIZE] = {3, 2, 1, 1, 4, 6, 5, 9, 8, 0};

// Function prototype
void show_array(void);
int get_mid(int x, int y, int z);
void bubblesort(void);

// Main Function
int main(void) {
    std::cout << "START" << std::endl;
    show_array();

    bubblesort();

    std::cout << "END" << std::endl;
    show_array();
    return 0;
}

void show_array(void) {
    std::cout << "=> ";
    for (int i = 0; i < SIZE; i++) {
        std::cout << array[i] << ", ";
    }

    std::cout << std::endl;
}

void bubblesort(void) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = SIZE-1; i < j; j--){
            if (array[j] < array[j-1]) {
                // exchange
                int temp   = array[j];
                array[j]   = array[j-1];
                array[j-1] = temp;
            }
        }
    }
}
