#include <stdio.h>
#include <time.h>
#define PR_DATE printf("Now time is %ld\n", time(NULL))

int main()
{
#ifdef PR_DATE 
    PR_DATE;
#else
    printf("Undefine PR_DATE!!\n");
#endif

    return 0;
}