#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <mqueue.h>
#include <fcntl.h> /* For definition of O_NONBLOCK */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"       // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

static void notifySetup(mqd_t *mqdp);

/* Thread notification function */
/**
 * 由核心自動建立的執行緒進入此函式。
 * 先呼叫 mq_getattr() 取得佇列屬性，配置 buffer。
 * 再次呼叫 notifySetup() → 因為 POSIX 規範要求通知只會觸發一次，必須重新註冊。
 * 使用 mq_receive() 讀取所有訊息並印出。
 */
static void threadFunc(union sigval sv) {
    ssize_t numRead;
    mqd_t *mqdp;
    void *buffer;
    struct mq_attr attr;

    mqdp = sv.sival_ptr;
    if (mq_getattr(*mqdp, &attr) == -1)
        errExit("mq_getattr");

    buffer = malloc(attr.mq_msgsize);
    if (buffer == NULL)
        errExit("malloc");

    notifySetup(mqdp);
    while ((numRead = mq_receive(*mqdp, buffer, attr.mq_msgsize, NULL)) >= 0)
        printf("Read %ld bytes\n", (long) numRead);

    if (errno != EAGAIN) /* Unexpected error */
        errExit("mq_receive");

    free(buffer);
    pthread_exit(NULL);
}

/**
 * 呼叫 notifySetup()，在裡面建立 struct sigevent 並指定：
 * sev.sigev_notify = SIGEV_THREAD;
 * sev.sigev_notify_function = threadFunc;
 * sev.sigev_value.sival_ptr = mqdp;
 * 這表示：當佇列有新訊息到達時，核心會建立一個新執行緒，並呼叫 threadFunc()，同時把 mqdp 傳入。
 */
static void notifySetup(mqd_t *mqdp) {
    struct sigevent sev;
    sev.sigev_notify = SIGEV_THREAD; /* Notify via thread */
    sev.sigev_notify_function = threadFunc;
    sev.sigev_notify_attributes = NULL;

    /* Could be pointer to pthread_attr_t structure */

    sev.sigev_value.sival_ptr = mqdp; /* Argument to threadFunc() */
    if (mq_notify(*mqdp, &sev) == -1)
        errExit("mq_notify");
}

int main(int argc, char *argv[])
{
    mqd_t mqd;
    if (argc != 2 || strcmp(argv[1], "--help") == 0)
        usageErr("%s mq-name\n", argv[0]);

    // 開啟佇列
    mqd = mq_open(argv[1], O_RDONLY | O_NONBLOCK);
    if (mqd == (mqd_t) -1)
        errExit("mq_open");
    
    // 設定通知
    notifySetup(&mqd);
    
    pause(); /* Wait for notifications via thread function */
}
