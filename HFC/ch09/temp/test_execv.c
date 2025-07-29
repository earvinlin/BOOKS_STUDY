#include <unistd.h>
#include <stdio.h>

int main() 
{
    char *args[] = { "ls", "-l", "/home", NULL };
    execv("/bin/ls", args);

    // 如果 execv 成功，這行不會被執行
    perror("execv failed");
    return 1;
}

