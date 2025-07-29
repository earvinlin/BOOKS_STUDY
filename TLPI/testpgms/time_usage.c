#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char* now()
{
    time_t t;
    time (&t);
    
    return asctime(localtime (&t));
}

/* Master Control Program utility.
Records guard patrol check-ins. */
int main()
{
    char comment[80];
    char cmd[120];


    system(cmd);
    
    return 0;
}
