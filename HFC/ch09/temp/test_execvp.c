#include <unistd.h>
#include <stdio.h>

int main() 
{
    char *args[] = { "ls", "-l", "/home", NULL };
    execvp("ls", args);

    // 如果 execvp 成功，這行不會被執行
    perror("execvp failed");
    return 1;
}

