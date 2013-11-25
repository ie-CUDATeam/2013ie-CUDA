#include <iostream>
#define SIZE 10 

int array[SIZE] =  {3, 2, 1, 1, 4, 6, 5, 9, 8, 0};

void bucket_sort(void);
void show_array(void);
int main() {
  std::cout << "START" << std::endl;
  show_array();

  bucket_sort();

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


void bucket_sort(void) {
  int i, j, k = 0;
  int count[SIZE];

  for(i = 0; i < SIZE; i++) count[i] = 0;
  for(i = 0; i < SIZE; i++) count[array[i]]++;
  for(i = 0; i < SIZE; i++) {
    for(j = count[i]; j > 0; j--) array[k++] = i;
  }
}
