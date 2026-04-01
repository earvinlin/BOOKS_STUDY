#include <signal.h>
#include "curr_time.h" /* Declaration of currTime() */
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define SYNC_SIG SIGUSR1 /* Synchronization signal */

/* Signal handler - does nothing but return */
static void handler(int sig) {
}

int main(int argc, char *argv[])
{
    pid_t childPid;
    sigset_t blockMask, origMask, emptyMask;
    struct sigaction sa;

    // 阻塞訊號
    setbuf(stdout, NULL); /* Disable buffering of stdout */
    sigemptyset(&blockMask);
    sigaddset(&blockMask, SYNC_SIG); /* Block signal */

    if (sigprocmask(SIG_BLOCK, &blockMask, &origMask) == -1)
        errExit("sigprocmask");

    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sa.sa_handler = handler;    // 設定訊號處理器

    if (sigaction(SYNC_SIG, &sa, NULL) == -1)
        errExit("sigaction");

    switch (childPid = fork()) {
        case -1:
            errExit("fork");
        case 0: /* Child */
            /* Child does some required action here... */
            printf("[%s %ld] Child started - doing some work\n",
            currTime("%T"), (long) getpid());
            sleep(2); /* Simulate time spent doing some work */

            /* And then signals parent that it's done */
            printf("[%s %ld] Child about to signal parent\n",
            currTime("%T"), (long) getpid());
            if (kill(getppid(), SYNC_SIG) == -1)
                errExit("kill");

            /* Now child can do other things... */
            _exit(EXIT_SUCCESS);
        default: /* Parent */
            /* Parent may do some work here, and then waits for child to
                complete the required action */
            printf("[%s %ld] Parent about to wait for signal\n",
            currTime("%T"), (long) getpid());
            sigemptyset(&emptyMask);

            if (sigsuspend(&emptyMask) == -1 && errno != EINTR)
                errExit("sigsuspend");

            printf("[%s %ld] Parent got signal\n", currTime("%T"), (long) getpid());

            /* If required, return signal mask to its original state */
            if (sigprocmask(SIG_SETMASK, &origMask, NULL) == -1)
                errExit("sigprocmask");

            /* Parent carries on to do other things... */
            exit(EXIT_SUCCESS);
    }   
}
