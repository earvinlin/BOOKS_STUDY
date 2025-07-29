/**
 * 展示如何使用 pipe()、fork() 和 exec() 來模擬 shell 指令 ls | grep txt。
 * 這個指令會列出目前目錄下所有檔案，並過濾出包含 txt 的檔案名稱。
 * 元件          功能說明
 * ===================================================
 * pipe()       建立一個無名管道，連接 ls 和 grep
 * fork()       建立兩個子進程，分別執行 ls 和 grep
 * dup2()       重定向標準輸入/輸出到管道
 * execlp()     執行外部指令，取代原進程映像
 * waitpid()    等待子進程結束，避免殭屍進程
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    int pipefd[2];
    pid_t pid1, pid2;

    // 建立管道
    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    // 建立第一個子進程：執行 ls
    pid1 = fork();
    if (pid1 == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid1 == 0) {
        // 子進程1：ls
        close(pipefd[0]);               // 關閉讀取端
        dup2(pipefd[1], STDOUT_FILENO); // 將標準輸出導向管道寫入端
        close(pipefd[1]);               // 關閉原始寫入端

        execlp("ls", "ls", NULL);       // 執行 ls
        perror("execlp ls");            // 若失敗
        exit(EXIT_FAILURE);
    }

    // 建立第二個子進程：執行 grep txt
    pid2 = fork();
    if (pid2 == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid2 == 0) {
        // 子進程2：grep txt
        close(pipefd[1]);               // 關閉寫入端
        dup2(pipefd[0], STDIN_FILENO);  // 將標準輸入導向管道讀取端
        close(pipefd[0]);               // 關閉原始讀取端

        execlp("grep", "grep", "txt", NULL); // 執行 grep txt
        perror("execlp grep");               // 若失敗
        exit(EXIT_FAILURE);
    }

    // 父進程：關閉管道並等待子進程結束
    close(pipefd[0]);
    close(pipefd[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);

    return 0;
}
