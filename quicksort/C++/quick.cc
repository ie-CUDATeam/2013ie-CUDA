#include <iostream>

// INPUT
const int SIZE  = 10;
int array[SIZE] = {3, 2, 1, 1, 4, 6, 5, 9, 8, 0};

// Function prototype
void show_array(void);
int get_mid(int x, int y, int z);
void quicksort(int left, int right);

// Main Function
int main(void) {
    std::cout << "START" << std::endl;
    show_array();

    int array_start = 0;
    int array_end   = SIZE - 1;
    quicksort(array_start, array_end);

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

/*
 * return mid value of 3 argument 
 * sample
 *   input (2,4,1)
 *   return 2
 */
int get_mid(int x, int y, int z) {
    if ((y<=x&&x<=z) || (z<=x&&x<=y)) {
        return x; 
    }
    if ((x<=y&&y<=z) || (z<=y&&y<=x)) {
        return y; 
    }
    return z;
}

/*
 * left  : sort start position
 * right : sort end position
 * */
void quicksort(int left, int right) {
    if (right <= left ) { 
        return; 
    }

    int i = left;
    int j = right;

    int x = array[i];
    int y = array[i+(j-i)/2];
    int z = array[j];
    int pivot = get_mid(x, y, z);
    
    while (true) {
        // find
        while (array[i] < pivot) i++; // find i
        while (pivot < array[j]) j--; // find j

        if (j <= i) break;

        // exchange
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;

        i++;
        j--;
    }

    quicksort(left, i-1); 
    quicksort(j+1, right); 
}
