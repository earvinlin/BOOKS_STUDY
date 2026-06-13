#include <signal.h>
#include <syslog.h>
#include <sys/wait.h>
#include "../../../tlpi-book/mylib/become_daemon.h"
#include "../../../tlpi-book/mylib/inet_sockets.h" /* Declarations of inet*() socket functions */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define SERVICE "echo" /* Name of TCP service */
#define BUF_SIZE 4096

/* SIGCHLD handler to reap dead child processes */
static void grimReaper(int sig) {
    int savedErrno; /* Save 'errno' in case changed here */
    savedErrno = errno;

    while (waitpid(-1, NULL, WNOHANG) > 0)
        continue;

        errno = savedErrno;
}

/* Handle a client request: copy socket input back to socket */
static void handleRequest(int cfd) {
    char buf[BUF_SIZE];
    ssize_t numRead;

    // 6. Echo 功能
    // 子行程讀取 client 傳來的資料，再原封不動回送給 client。
    // 若 read() 或 write() 出錯，記錄 syslog 並結束。
    while ((numRead = read(cfd, buf, BUF_SIZE)) > 0) {
        if (write(cfd, buf, numRead) != numRead) {
            syslog(LOG_ERR, "write() failed: %s", strerror(errno));
            exit(EXIT_FAILURE);
        }
    }

    if (numRead == -1) {
        syslog(LOG_ERR, "Error from read(): %s", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    int lfd, cfd; /* Listening and connected sockets */
    struct sigaction sa;

    // 1. Daemon 化
    if (becomeDaemon(0) == -1)
        errExit("becomeDaemon");

    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    // 2. SIGCHLD 處理
    // 設定 grimReaper() 作為 SIGCHLD handler。
    // 當子行程結束時，父行程會呼叫 waitpid(-1, NULL, WNOHANG) 清理殭屍行程，避免資源浪費。
    sa.sa_handler = grimReaper;
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        syslog(LOG_ERR, "Error from sigaction(): %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // 3. 建立 TCP 監聽 socket
    // 使用 inetListen() 建立 TCP socket，綁定到服務名稱 "echo"，並設定 backlog = 10。
    // 若失敗，透過 syslog(LOG_ERR, ...) 記錄錯誤。
    lfd = inetListen(SERVICE, 10, NULL);
    if (lfd == -1) {
        syslog(LOG_ERR, "Could not create server socket (%s)", strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (;;) {
        // 4. 主迴圈：接受連線
        // 等待 client 連線，成功後回傳 connected socket cfd。
        cfd = accept(lfd, NULL, NULL); /* Wait for connection */
        if (cfd == -1) {
            syslog(LOG_ERR, "Failure in accept(): %s", strerror(errno));
            exit(EXIT_FAILURE);
        }

        /* Handle each client request in a new child process */

        // 5. fork 子行程處理 client
        switch (fork()) {
            case -1:
                syslog(LOG_ERR, "Can't create child (%s)", strerror(errno));
                close(cfd); /* Give up on this client */
                break; /* May be temporary; try next client */
            case 0: /* Child */
                // 子行程：關閉 lfd（不需要監聽 socket），呼叫 handleRequest() 處理 client。
                close(lfd); /* Unneeded copy of listening socket */
                handleRequest(cfd);
                _exit(EXIT_SUCCESS);
            default: /* Parent */
                // 父行程：關閉 cfd，繼續等待下一個 client。
                close(cfd); /* Unneeded copy of connected socket */
                break; /* Loop to accept next connection */
        }
    }
}
