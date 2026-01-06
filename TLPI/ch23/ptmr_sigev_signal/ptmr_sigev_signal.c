#define _POSIX_C_SOURCE 199309
#include <signal.h>
#include <time.h>
#include "curr_time.h" /* Declares currTime() */
#include "itimerspec_from_str.h" /* Declares itimerspecFromStr() */
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define TIMER_SIG SIGRTMAX /* Our timer notification signal */

// (1)訊號處理函式
static void handler(int sig, siginfo_t *si, void *uc) {
    timer_t *tidptr;
    // 使用 siginfo_t 取得 sigev_value.sival_ptr，知道是哪個定時器觸發
    tidptr = si->si_value.sival_ptr;    

    /* UNSAFE: This handler uses non-async-signal-safe functions
        (printf(); see Section 21.1.2) */
    printf("[%s] Got signal %d\n", currTime("%T"), sig);
    printf(" *sival_ptr = %ld\n", (long) *tidptr);
    // timer_getoverrun()：檢查是否有訊號遺失(定時器過期次數超過一次)
    printf(" timer_getoverrun() = %d\n", timer_getoverrun(*tidptr));
}

int main(int argc, char *argv[])
{
    struct itimerspec ts;
    struct sigaction sa;
    struct sigevent sev;
    timer_t *tidlist;
    int j;

    if (argc < 2)
        usageErr("%s secs[/nsecs][:int-secs[/int-nsecs]]...\n", argv[0]);
        
    tidlist = calloc(argc - 1, sizeof(timer_t));
    if (tidlist == NULL)
        errExit("malloc");

    /* 建立訊號處理器 : Establish handler for notification signal */
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = handler;
    sigemptyset(&sa.sa_mask);
    // (2)建立用於計時器通知的訊號處理常式
    if (sigaction(TIMER_SIG, &sa, NULL) == -1)
        errExit("sigaction");

    /* 建立定時器 : Create and start one timer for each command-line argument */
    sev.sigev_notify = SIGEV_SIGNAL; /* Notify via signal */
    sev.sigev_signo = TIMER_SIG; /* Notify using this signal */
    for (j = 0; j < argc - 1; j++) {
        // (3)
        itimerspecFromStr(argv[j + 1], &ts);
        sev.sigev_value.sival_ptr = &tidlist[j];

        /* (4)Allows handler to get ID of this timer */
        if (timer_create(CLOCK_REALTIME, &sev, &tidlist[j]) == -1)
            errExit("timer_create");
    
        printf("Timer ID: %ld (%s)\n", (long) tidlist[j], argv[j + 1]);
        // (5)
        if (timer_settime(tidlist[j], 0, &ts, NULL) == -1)
            errExit("timer_settime");
    }

    /* (6)等待訊號 : Wait for incoming timer signals */
    for (;;) 
        pause();
}
