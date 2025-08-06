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
//    system("ls -l ~/");
    char comment[80];
    char cmd[120];

    fgets(comment, 80, stdin);
    sprintf(cmd, "echo '%s %s' >> reports.log", comment, now());

    system(cmd);

    return 0;
}