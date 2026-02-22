#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

int main() {
    pid_t pgid = 0;

    for (int i = 0; i < 3; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // child
            if (i == 0) {
                // 第一個 worker 成為 group leader
                setpgid(0, 0);
            } else {
                // 其他 worker 加入同一個 group
                setpgid(0, pgid);
            }

            printf("Worker PID=%d, PGID=%d\n", getpid(), getpgrp());
            sleep(10);
            return 0;
        } else {
            // parent
            if (i == 0) {
                pgid = pid;  // 第一個 child 的 pid 當作 pgid
            }
        }
    }

    sleep(2);
    printf("一次 kill 整組 workers (PGID=%d)\n", pgid);
    killpg(pgid, SIGTERM);

    while (wait(NULL) > 0);
    printf("全部 worker 已結束\n");

    return 0;
}