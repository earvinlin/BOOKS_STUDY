#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NUM_1 10
#define NUM_2 30

void fillarray(int ar[], int n);
void showarray(const int ar[], int n);
int mycomp(const void * p1, const void * p2);

int main(void) 
{
    int vals_100[NUM_1];
    int vals_300[NUM_2];

    fillarray(vals_100, NUM_1);
    fillarray(vals_300, NUM_2);

    puts("Random list:");
    showarray(vals_100, NUM_1);
    printf("\n");
    showarray(vals_300, NUM_2);

    // 將vals_300前10個元素複製到vals_100
//    memcpy(vals_100, vals_300, 10*sizeof(int));

    // 將vals_300中間10個元素複製到vals_100
//    memcpy(vals_100, &vals_300[10], 10*sizeof(int));
    memcpy(vals_100, vals_300+10, 10*sizeof(int));
    
    puts("\nAfter copy:");
    showarray(vals_100, NUM_1);
    printf("\n");
    showarray(vals_300, NUM_2);
    
    return 0;
}

void fillarray(int ar[], int n) {
    int index;
    for( index = 0; index < n; index++)
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

