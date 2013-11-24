/*
 * mergesort.c
 *
 *  Created on: Nov 23, 2013
 *      Author: Noble713
 */

#include<stdio.h>
#include<conio.h>

const int size  = 10;
int a[10] = {3, 2, 1, 1, 4, 6, 5, 9, 8, 0};

void mergesort (void);

int main()
{

 int g;
 printf("\n\t------- Merge sorting method, unsorted array -------\n\n");
 for(g=0; g<size; g++)
 printf("%d ",a[g]);  /* displays unsorted values*/

 mergesort();
 printf("\n\t------- Merge sorted elements -------\n\n");
 for(g=0; g<size; g++)
 printf("%d ",a[g]);
 getch();
 return 0;
}


void mergesort (void)
{
	int b[size];
    int right, rend;
    int i,j,k,m;
    int left =0;

    for (k=1; k < size; k *= 2 ) {
        for (left=0; left+k < size; left += k*2 ) {
            right = left + k;
            rend = right + k;
            if (rend > size) rend = size;
            m = left; i = left; j = right;
            while (i < right && j < rend) {
                if (a[i] <= a[j]) {
                    b[m] = a[i]; i++;
                } else {
                    b[m] = a[j]; j++;
                }
                m++;
            }
            while (i < right) {
                b[m]=a[i];
                i++; m++;
            }
            while (j < rend) {
                b[m]=a[j];
                j++; m++;
            }
            for (m=left; m < rend; m++) {
                a[m] = b[m];
            }
        }
    }
}



