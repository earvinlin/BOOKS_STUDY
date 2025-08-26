#include <stdio.h>
#define NUM25 25
#define SPACE " "
#define PS() printf(" ")
#define BIG(X) (3 + X)
#define SUMSQ(X, Y) ((X*X) + (Y*Y))

int main()
{
    printf("NUM25 is %d\n", NUM25);
    printf("PRINT SPACE ->" SPACE "<-\n");
    printf("Pring Space fun PS() ->");
    PS();
    printf("<-\n");
    printf("BIG(5)+3 is %d\n", BIG(5));
    printf("SUMSQ(2,3) is %d\n", SUMSQ(2, 3));

    return 0;
}