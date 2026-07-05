#include <sys/time.h>
#include <sys/select.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

static void usageError(const char *progName) {
    fprintf(stderr, "Usage: %s {timeout|-} fd-num[rw]...\n", progName);
    fprintf(stderr, " - means infinite timeout; \n");
    fprintf(stderr, " r = monitor for read\n");
    fprintf(stderr, " w = monitor for write\n\n");
    fprintf(stderr, " e.g.: %s - 0rw 1w\n", progName);

    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    fd_set readfds, writefds;
    int ready, nfds, fd, numRead, j;
    struct timeval timeout;
    struct timeval *pto;
    char buf[10]; /* Large enough to hold "rw\0" */

    if (argc < 2 || strcmp(argv[1], "--help") == 0)
        usageError(argv[0]);

    /* Timeout for select() is specified in argv[1] */

    // 程式先處理第一個參數 argv[1]，用來設定 select() 要等多久
    if (strcmp(argv[1], "-") == 0) {
        pto = NULL; /* Infinite timeout */
    } else {
        pto = &timeout;
        timeout.tv_sec = getLong(argv[1], 0, "timeout");
        timeout.tv_usec = 0; /* No microseconds */
    }

    /* Process remaining arguments to build file descriptor sets */

    // 核心：建立點名簿（Building File Descriptor Sets）
    nfds = 0;
    FD_ZERO(&readfds);
    FD_ZERO(&writefds);

    for (j = 2; j < argc; j++) {
        // 使用 sscanf(argv[j], "%d%2[rw]", &fd, buf)。如果使用者輸入 0rw，
        // 它會把 0 抓進 fd 變數，把 "rw" 抓進 buf 陣列。
        numRead = sscanf(argv[j], "%d%2[rw]", &fd, buf);
        
        if (numRead != 2)
            usageError(argv[0]);
        if (fd >= FD_SETSIZE)
            cmdLineErr("file descriptor exceeds limit (%d)\n", FD_SETSIZE);
        
        // 計算上限 nfds：這是上一題提到的效能優化關鍵。每次發現更大的 fd，就更新 
        // nfds = fd + 1。迴圈結束後，nfds 就會是所有監聽 fd 中最大值再加 1。
        if (fd >= nfds)
            nfds = fd + 1; /* Record maximum fd + 1 */
        // 如果字串裡有 'r'，就用 FD_SET(fd, &readfds) 勾選。
        if (strchr(buf, 'r') != NULL)
            FD_SET(fd, &readfds);
        // 如果字串裡有 'w'，就用 FD_SET(fd, &writefds) 勾選。
        if (strchr(buf, 'w') != NULL)
            FD_SET(fd, &writefds);
    }

    /* We've built all of the arguments; now call select() */

    // 呼叫系統並等待（Calling select）
    // ．注意第四個參數：程式傳入了 NULL，這呼應了前一題所說的——因為一般的程式不需要
    //   捕捉頻外資料或虛擬終端機狀態改變，所以直接忽略例外監聽（exceptfds）。
    // ．阻塞等待：此時程式會卡在這一行。當指定的 fd 發生動靜，或者時間到，核心就會修
    //   改 readfds、writefds 與 timeout 的內容，然後解除阻塞，回傳就緒的 fd 總
    //   數給 ready。
    ready = select(nfds, &readfds, &writefds, NULL, pto);

    /* Ignore exceptional events */
    if (ready == -1)
        errExit("select");

    /* Display results of select() */
    // 結果呈現與驗證（Display Results）
    // ．印出總數：告訴你共有幾個事件就緒（ready）。
    // ．逐一檢查：利用迴圈從 0 掃描到 nfds - 1，並用 FD_ISSET 去看哪一個 fd 還被核
    //   心留在名單上。如果留在 readfds 就印出 r，留在 writefds 就印出 w。
    // ．觀察 Linux 獨有行為：最後，如果當初有設定倒數時間（pto != NULL），程式會把呼
    //   叫後的 timeout 剩餘時間印出來。這正是上一題提到的 Linux 獨特行為——如果時間還
    //   沒到就有 fd 觸發，你可以透過這個 printf 看到計時器還剩下多少秒。
    printf("ready = %d\n", ready);
    for (fd = 0; fd < nfds; fd++)
        printf("%d: %s%s\n", fd, FD_ISSET(fd, &readfds) ? "r" : "",
            FD_ISSET(fd, &writefds) ? "w" : "");
    
    if (pto != NULL)
        printf("timeout after select(): %ld.%03ld\n",
            (long) timeout.tv_sec, (long) timeout.tv_usec / 10000);

    exit(EXIT_SUCCESS);
}
