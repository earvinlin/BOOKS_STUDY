#include <unistd.h>
#include <stdio.h>

int main() {
    execlp("open", "open", "-a", "Calculator", NULL);

    // 如果 execlp 成功，這行不會被執行
    perror("execlp failed");
    return 1;
}

