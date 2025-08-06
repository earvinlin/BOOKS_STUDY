#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 回傳字串，包含當前的資料與時間
char* now()
{
    time_t t;
    time (&t);
    return asctime(localtime (&t));
}

int main()
{
    
    return 0;
}