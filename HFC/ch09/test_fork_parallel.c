#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

void taskA() {
    printf("[子行程 A] PID: %d 正在執行計算任務...\n", getpid());
    for (int i = 0; i < 5; i++) {
        printf("[子行程 A] 計算中：%d\n", i * i);
        sleep(1);
    }
    printf("[子行程 A] 任務完成。\n");
}

void taskB() {
    printf("[子行程 B] PID: %d 正在執行 I/O 模擬...\n", getpid());
    for (int i = 0; i < 3; i++) {
        printf("[子行程 B] 模擬寫入資料 %d\n", i);
        sleep(2);
    }
    printf("[子行程 B] 任務完成。\n");
}

int main() {
    pid_t pidA, pidB;

    pidA = fork();
    if (pidA == 0) {
        taskA();
        exit(0);
    }

    pidB = fork();
    if (pidB == 0) {
        taskB();
        exit(0);
    }

    // 父行程
    printf("[父行程] PID: %d 正在監控子行程...\n", getpid());

    // 等待子行程結束
    waitpid(pidA, NULL, 0);
    waitpid(pidB, NULL, 0);

    printf("[父行程] 所有子行程已完成。\n");
    return 0;
}

