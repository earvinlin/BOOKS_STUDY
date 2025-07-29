#include <unistd.h>
#include <stdio.h>

int main() {
    execlp("mate-calc", "mate-calc", NULL);

    // 如果 execlp 成功，這行不會被執行
    perror("execlp failed");
    return 1;
}

