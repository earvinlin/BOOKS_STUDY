#include <signal.h>
#include <mqueue.h>
#include <fcntl.h> /* For definition of O_NONBLOCK */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"       // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define NOTIFY_SIG SIGUSR1

static void handler(int sig) {
    /* Just interrupt sigsuspend() */
}

int main(int argc, char *argv[])
{
    struct sigevent sev;
    mqd_t mqd;
    struct mq_attr attr;
    void *buffer;
    ssize_t numRead;
    sigset_t blockMask, emptyMask;
    struct sigaction sa;

    if (argc != 2 || strcmp(argv[1], "--help") == 0)
        usageErr("%s mq-name\n", argv[0]);

    mqd = mq_open(argv[1], O_RDONLY | O_NONBLOCK);
    if (mqd == (mqd_t) -1)
        errExit("mq_open");
    if (mq_getattr(mqd, &attr) == -1)
        errExit("mq_getattr");

    buffer = malloc(attr.mq_msgsize);
    if (buffer == NULL)
        errExit("malloc");

    // 設定訊號處理
    // 先阻塞通知訊號 (SIGUSR1)，避免在設定完成前被觸發。
    // 設定 sigaction()，指定 handler 函式。
    sigemptyset(&blockMask);
    sigaddset(&blockMask, NOTIFY_SIG);
    if (sigprocmask(SIG_BLOCK, &blockMask, NULL) == -1)
        errExit("sigprocmask");

    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sa.sa_handler = handler;

    if (sigaction(NOTIFY_SIG, &sa, NULL) == -1)
        errExit("sigaction");
    
    // 註冊通知 : 告訴核心：當佇列有新訊息到達時，送出 SIGUSR1 給此行程。
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = NOTIFY_SIG;
    if (mq_notify(mqd, &sev) == -1)
        errExit("mq_notify");

    sigemptyset(&emptyMask);
    for (;;) {
        // 等待通知
        // 暫時解除阻塞，等待訊號到來。
        // 收到 SIGUSR1 後，handler 被呼叫，sigsuspend() 返回。
        sigsuspend(&emptyMask); /* Wait for notification signal */
        // 重新註冊通知 : POSIX 規範，通知只會觸發一次，必須在每次收到訊號後重新註冊。
        if (mq_notify(mqd, &sev) == -1)
            errExit("mq_notify");

        // 讀取訊息
        // 迴圈讀取佇列中的所有訊息。
        // 若佇列空，mq_receive() 回傳 -1 並設 errno = EAGAIN。
        while ((numRead = mq_receive(mqd, buffer, attr.mq_msgsize, NULL)) >= 0)
            printf("Read %ld bytes\n", (long) numRead);
        if (errno != EAGAIN) /* Unexpected error */
            errExit("mq_receive");
    }
}
