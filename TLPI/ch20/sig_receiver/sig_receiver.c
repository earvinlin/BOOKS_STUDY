#define _GNU_SOURCE
#include <signal.h>
#include "../../tlpi-book/signals/signal_functions.h" /* Declaration of printSigset() */
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

static int sigCnt[NSIG]; /* Counts deliveries of each signal NSIG 是系統支援的訊號數量 (max=64) */
static volatile sig_atomic_t gotSigint = 0;

/* Set nonzero if SIGINT is delivered [CTRL+C] */
static void handler(int sig) {
    if (sig == SIGINT)
        gotSigint = 1;
    else
        sigCnt[sig]++;
}

int main(int argc, char *argv[])
{
    int n, numSecs;
    sigset_t pendingMask, blockingMask, emptyMask;
    printf("%s: PID is %ld\n", argv[0], (long) getpid());

    for (n = 1; n < NSIG; n++) /* Same handler for all signals */
        (void) signal(n, handler); /* Ignore errors */

    /* If a sleep time was specified, temporarily block all signals,
        sleep (while another process sends us signals), and then
        display the mask of pending signals and unblock all signals */

    if (argc > 1) {
        numSecs = getInt(argv[1], GN_GT_0, NULL);
    
        // 建立一個包含所有訊號的集合
        sigfillset(&blockingMask);

        // 設定目前遮罩為「阻塞所有訊號」；睡眠期間，訊號會進入「擱置狀態」
        if (sigprocmask(SIG_SETMASK, &blockingMask, NULL) == -1)
            errExit("sigprocmask");

        printf("%s: sleeping for %d seconds\n", argv[0], numSecs);
        sleep(numSecs);

        // 顯示擱置訊號
        // sigpending() : 取得目前擱置的訊號集合
        if (sigpending(&pendingMask) == -1)
            errExit("sigpending");

        printf("%s: pending signals are: \n", argv[0]);
        // 列印集合內容（TLPI 自訂函式）
        printSigset(stdout, "\t\t", &pendingMask);
        sigemptyset(&emptyMask); /* Unblock all signals */

        if (sigprocmask(SIG_SETMASK, &emptyMask, NULL) == -1)
            errExit("sigprocmask");
    }

    while (!gotSigint) /* Loop until SIGINT caught */
        continue;

    // 顯示訊號接收次數
    for (n = 1; n < NSIG; n++) /* Display number of signals received */
        if (sigCnt[n] != 0)
            printf("%s: signal %d caught %d time%s\n", argv[0], n,
                sigCnt[n], (sigCnt[n] == 1) ? "" : "s");

    exit(EXIT_SUCCESS);
}
/**
 * 學到的重點
 * signal() 安裝處理器。
 * sigprocmask() 控制訊號阻塞。
 * sigpending() 檢查擱置訊號。
 * sigfillset() / sigemptyset() 操作訊號集合。
 * 解除阻塞後，擱置訊號會立即處理。
 */