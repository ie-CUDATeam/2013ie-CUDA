#include <stdio.h>

#define MAX_DATA 10

int temp[MAX_DATA];    /* 最小でも配列と同じサイズの領域が必要 */

void MergeSort(int x[ ], int left, int right);

/* 配列 x[ ] の left から right の要素のマージソートを行う */
void MergeSort(int x[ ], int left, int right)
{
  int mid, i, j, k;

  if (left >= right)              /* 配列の要素がひとつなら */
    return;                     /* 何もしないで戻る */

  /* ここでは分割しているだけ */
  mid = (left + right) / 2;       /* 中央の値より */
  MergeSort(x, left, mid);        /* 左を再帰呼び出し */
  MergeSort(x, mid + 1, right);   /* 右を再帰呼び出し */

  /* x[left] から x[mid] を作業領域にコピー */
  for (i = left; i <= mid; i++)
    temp[i] = x[i];

  /* x[mid + 1] から x[right] は逆順にコピー */
  for (i = mid + 1, j = right; i <= right; i++, j--)
    temp[i] = x[j];

  i = left;         /* i とj は作業領域のデーターを */
  j = right;        /* k は配列の要素を指している */

  for (k = left; k <= right; k++)    /* 小さい方から配列に戻す */
    if (temp[i] <= temp[j])        /* ここでソートされる */
      x[k] = temp[i++];
    else
      x[k] = temp[j--];
}

int main(void)
{
  int i;
  /* ソートされるデータ */
  int x[ ] = {9, 4, 6, 2, 1, 8, 0, 3, 7, 5};

  /* ソート前のデータを表示 */
  printf("ソート前\n");
  for (i = 0; i < MAX_DATA; i++)
    printf("%d\t", x[i]);

  MergeSort(x, 0, MAX_DATA - 1);

  /* ソート後のデータを表示 */
  printf("ソート後\n");
  for (i = 0; i < MAX_DATA; i++)
    printf("%d\t", x[i]);
  return(0);
}
