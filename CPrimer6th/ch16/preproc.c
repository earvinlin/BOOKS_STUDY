/* 
 * 範例 16.1 (p.16-2)
 * preproc.c -- simple preprocessor examples 
 */
#include <stdio.h>
#define TWO 2 /* you can use comments if you like */
#define OW "Consistency is the last refuge of the unimagina\
tive. - Oscar Wilde" /* a backslash continues a definition */
/* to the next line */
#define FOUR TWO*TWO
#define PX printf("X is %d.\n", x)
#define FMT "X is %d.\n"

#define MEAN(X, Y) (((X) + (Y)) / 2)
#define SQUARE(X) X*X

int main(void)
{
    int x = TWO;
    PX;
    x = FOUR;
    printf(FMT, x);
    printf("%s\n", OW);
    printf("TWO: OW\n");
    
    printf("55+32 = %d\n", MEAN(55, 32));       // ok
    printf("55+32 = %f\n", MEAN(55, 32));       // 計算錯誤
    printf("55.1+32 = %f\n", MEAN(55.1, 32));   // ok
    printf("55.1+32 = %d\n", MEAN(55.1, 32));   // 計算錯誤

    float s = MEAN(55,32);
    printf("s(55+32) = %f\n", s);   // ok

    

    printf("%d\n", SQUARE( 3 ));
    printf("%f\n", SQUARE( 3.1 ));
    float z = SQUARE( 3.1 );
    printf("%f\n", z);

    return 0;
}

