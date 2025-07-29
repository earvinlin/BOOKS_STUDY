#include <unistd.h>
#include <stdio.h>

int main() 
{
    execl("/bin/ls", "ls", "-l", "/home", NULL);

    // 如果 execl 成功，這行永遠不會執行
    perror("execl failed");
    return 1;
}

