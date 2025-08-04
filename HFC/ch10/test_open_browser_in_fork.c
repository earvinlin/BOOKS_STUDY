/**
 * 在C語言中，fork()是Linux/Unix系統提供的系統呼叫，用來建立一個新的子行程(process)。
 * 這個新行程是原本行程的幾乎完整複製，兩者會從fork()呼叫點開始平行執行。
 * pid_t fork(void) 
 *  <0  建立子行程失敗
 *  =0  子行程中執行
 *  >0  父行程中執行，回傳的是子行程的PID
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void open_url(char *url)
{
    char launch[255];
    // Windows
    sprintf(launch, "cmd /c start %s", url);
    system(launch);
    // Linux
    sprintf(launch, "x-www-browser '%s' &", url);
    system(launch);
    // Mac
    sprintf(launch, "open '%s'", url);
    system(launch);
}

int main(int argc, char *argv[]) 
{
    char *url = argv[1];
    open_url(url);

/**
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
    } else if (pid == 0) {
        printf("這是子行程，PID = %d，父 PID = %d\n", getpid(), getppid());
    } else {
        printf("這是父行程，PID = %d，子 PID = %d\n", getpid(), pid);
    } 
 */

    return 0;
}
