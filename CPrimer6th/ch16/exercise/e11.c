/* qsorter.c -- using qsort to sort groups of numbers */
#include <stdio.h>
#include <stdlib.h>
#define NUM 1000

void fillarray(int ar[], int n);
void showarray(const int ar[], int n);
int mycomp(const void * p1, const void * p2);

int main(void) 
{
    int vals[NUM];
    fillarray(vals, NUM);
    puts("Random list:");
    showarray(vals, NUM);
    qsort(vals, NUM, sizeof(int), mycomp);
    puts("\nSorted list:");
    showarray(vals, NUM);
    
    return 0;
}

void fillarray(int ar[], int n) {
    int index;
    for( index = 0; index < n; index++)
//        ar[index] = (double)rand()/((double) rand() + 0.1);
        ar[index] = rand();
}

void showarray(const int ar[], int n) {
    int index;

    for( index = 0; index < n; index++) {
        printf("%10d ", ar[index]);
        if (index % 6 == 5)
            putchar('\n');
    }

    if (index % 6 != 0)
        putchar('\n');
}

/* sort by increasing value */
int mycomp(const void * p1, const void * p2) {
    /* need to use pointers to double to access values */
    const int * a1 = (const int *) p1;
    const int * a2 = (const int *) p2;
/*
    if (*a1 < *a2)
        return -1;
    else if (*a1 == *a2)
        return 0;
    else
        return 1;
*/
    if (*a1 < *a2)
        return 1;
    else if (*a1 == *a2)
        return 0;
    else
        return -1;

}
