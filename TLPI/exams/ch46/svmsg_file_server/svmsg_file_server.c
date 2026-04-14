#include "../../../tlpi-book/svmsg/svmsg_file.h"

/* SIGCHLD handler */
static void grimReaper(int sig) {
    int savedErrno;

    savedErrno = errno; /* waitpid() might change 'errno' */

    // 使用 WNOHANG → 非阻塞，避免影響主程式
    while (waitpid(-1, NULL, WNOHANG) > 0)
        continue;

    errno = savedErrno;
}

/* Executed in child process: serve a single client */
// (5) serveRequest()：子行程讀檔案並回傳
static void serveRequest(const struct requestMsg *req) {
    int fd;
    ssize_t numRead;
    struct responseMsg resp;

    fd = open(req->pathname, O_RDONLY);
    if (fd == -1) { /* Open failed: send error text */
        resp.mtype = RESP_MT_FAILURE;
        snprintf(resp.data, sizeof(resp.data), "%s", "Couldn't open");
        msgsnd(req->clientId, &resp, strlen(resp.data) + 1, 0);
        exit(EXIT_FAILURE); /* and terminate */
    }
    
    /* Transmit file contents in messages with type RESP_MT_DATA. We don't
        diagnose read() and msgsnd() errors since we can't notify client. */
    //成功 → 回傳檔案內容
    // 每次讀到的資料用 message queue 傳回
    // clientId 是 client 自己的 queue
    // server → client 是 point-to-point
    resp.mtype = RESP_MT_DATA;  /* Message contains file data */
    while ((numRead = read(fd, resp.data, RESP_MSG_SIZE)) > 0)
        if (msgsnd(req->clientId, &resp, numRead, 0) == -1)
            break;
    
    /* Send a message of type RESP_MT_END to signify end-of-file */
    // 最後送 EOF 訊號
    resp.mtype = RESP_MT_END;   /* File data complete */
    msgsnd(req->clientId, &resp, 0, 0); /* Zero-length mtext */
}

int main(int argc, char *argv[])
{
    struct requestMsg req;
    pid_t pid;
    ssize_t msgLen;
    int serverId;
    struct sigaction sa;

    /* Create server message queue */

    // (1) 建立 Server Message Queue
    serverId = msgget(SERVER_KEY, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IWGRP);
    if (serverId == -1)
        errExit("msgget");

    /* Establish SIGCHLD handler to reap terminated children */

    // (2) 設定 SIGCHLD handler
    // 子行程結束時會送 SIGCHLD
    // grimReaper() 會回收子行程
    // SA_RESTART：若系統呼叫被中斷，自動重啟（例如 msgrcv）
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sa.sa_handler = grimReaper;

    if (sigaction(SIGCHLD, &sa, NULL) == -1)
        errExit("sigaction");

    /* Read requests, handle each in a separate child process */

    // (3) 主迴圈：等待 client request
    for (;;) {
        // 阻塞等待 client 傳 requestMsg
        // 若被 SIGCHLD 中斷 → errno = EINTR → 重新 msgrcv
        // 若其他錯誤 → 結束 server
        msgLen = msgrcv(serverId, &req, REQ_MSG_SIZE, 0, 0);
        if (msgLen == -1) {
            if (errno == EINTR) /* Interrupted by SIGCHLD handler? */
                continue; /* ... then restart msgrcv() */
            errMsg("msgrcv"); /* Some other error */
            break; /* ... so terminate loop */
        }
        // (4) fork 子行程處理 request
        pid = fork(); /* Create child process */
        if (pid == -1) {
            errMsg("fork");
            break;
        }
        
        if (pid == 0) { /* Child handles request */
            serveRequest(&req);
            _exit(EXIT_SUCCESS);
        }
        /* Parent loops to receive next client request */
    }

    /* If msgrcv() or fork() fails, remove server MQ and exit */
    // (6) server 結束時刪除 message queue
    if (msgctl(serverId, IPC_RMID, NULL) == -1)
        errExit("msgctl");

    exit(EXIT_SUCCESS);
}
