#define _XOPEN_SOURCE 500
#include <unistd.h>
#include <signal.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

// 定義一個空的訊號處理函式，當收到訊號時不做任何事。
// 這裡主要是用來攔截 `SIGHUP`，避免預設行為（通常是終止進程）。
static void handler(int sig) {
}

int main(int argc, char *argv[])
{
    pid_t childPid;
    struct sigaction sa;
    setbuf(stdout, NULL); /* Make stdout unbuffered */
    sigemptyset(&sa.sa_mask);

    sa.sa_flags = 0;
    sa.sa_handler = handler;

    // 設定 sigaction攔截 SIGHUP 訊號並交給 handler。
    // SIGHUP 常見於控制終端關閉或 session leader 結束時，
    // 會傳給同一 process group 的所有進程。
    if (sigaction(SIGHUP, &sa, NULL) == -1)
        errExit("sigaction");

    childPid = fork();

    if (childPid == -1)
        errExit("fork");

    // 如果是子進程（`childPid ==0）且命令列參數大於 1，
    // 則呼叫 setpgid(0,0) 把自己移到新的 process group。
    // 這樣子進程就不再跟父進程同一個 process group。
    // 這通常用來測試 session leader 與 process group 的行為。
    if (childPid == 0 && argc > 1)
        if (setpgid(0, 0) == -1) /* Move to new process group */
            errExit("setpgid");

    // PID：自己的進程 ID ; PPID：父進程 ID ;
    // PGID：所在的 process group ID ; SID：所在的 session ID
    printf("PID=%ld; PPID=%ld; PGID=%ld; SID=%ld\n", (long) getpid(),
        (long) getppid(), (long) getpgrp(), (long) getsid(0));
    alarm(60); /* An unhandled SIGALRM ensures this process
                  will die if nothing else terminates it */   

    for(;;) { /* Wait for signals */
        pause();    // 阻塞，直到收到訊號。
        printf("%ld: caught SIGHUP\n", (long) getpid());
    }
}
