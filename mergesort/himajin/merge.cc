#include<iostream>
using namespace std;
#define THREADS 1024
#define BLOCKS 1024
#define NUM_VALS THREADS*BLOCKS

void merge(int *arr, int size1, int size2) {
  int temp[size1+size2];
  int ptr1=0, ptr2=0;

  while (ptr1+ptr2 < size1+size2) {
    if (ptr1 < size1 && arr[ptr1] <= arr[size1+ptr2] || ptr1 < size1 && ptr2 >= size2)
      temp[ptr1+ptr2] = arr[ptr1++];

    if (ptr2 < size2 && arr[size1+ptr2] < arr[ptr1] || ptr2 < size2 && ptr1 >= size1)
      temp[ptr1+ptr2] = arr[size1+ptr2++];
  }

  for (int i=0; i < size1+size2; i++)
    arr[i] = temp[i];
}

void mergeSort(int *arr, int size) {
  if (size == 1) return;
  int size1 = size/2, size2 = size-size1;
  mergeSort(arr, size1);
  mergeSort(arr+size1, size2);

  merge(arr, size1, size2);
}

int randomInt() {
  return (int)(((float)rand()/(float)RAND_MAX) * 1000000);
}

void arrayFill(int *arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = randomInt();
  }
}

void arrayPrint(int *arr, int length) {
  int i;
  for (i = 0; i < length; ++i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}


int main(void) {
  int *values = (int*) malloc( NUM_VALS * sizeof(int));
  arrayFill(values, NUM_VALS);

  // Start merge sort
  mergeSort(values, NUM_VALS);

  // Print the sorted array
  // arrayPrint(values, NUM_VALS);
  return 0;
}
