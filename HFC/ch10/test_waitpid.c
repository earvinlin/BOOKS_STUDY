#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    if (pid == 0) {
        printf("In pid == 0\n");
        // 子進程
        sleep(2);
        exit(42);
    } else {
        printf("In else block, pid= %d\n", pid);
        // 父進程
        int status;
        pid_t result = waitpid(pid, &status, 0);
        if (result > 0 && WIFEXITED(status)) {
            printf("子進程正常退出，退出碼：%d\n", WEXITSTATUS(status));
        }
    }
    printf("return!!!\n");
    return 0;
}
