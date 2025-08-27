#include <stdio.h>
#define PRN_LINE 1

int main()
{
#ifdef PRN_LINE 
    printf("Define PRN_LINE!!\n");
#else
    printf("Undefine PRN_LINE!!\n");
#endif

    return 0;
}