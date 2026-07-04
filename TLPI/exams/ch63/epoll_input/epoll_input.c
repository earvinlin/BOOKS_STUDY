#include <sys/epoll.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define MAX_BUF 1000 /* Maximum bytes fetched by a single read() */
#define MAX_EVENTS 5 /* Maximum number of events to be returned from a single epoll_wait() call */

/*
    💡 總結這段程式碼的測試場景
    這個程式不適合拿來讀一般的硬碟檔案（因為一般檔案會立刻引發 EPOLLIN 讀完並結束）。
    它完美的測試場景是同時監控多個具名管線（FIFO）或終端機輸入。你可以開多個終端機視窗往不同的 FIFO 
    寫入資料，會看到這個程式只要哪邊有資料寫入，它就會立刻透過 epoll_wait() 醒來並精準印出是哪個 
    fd 收到幾位元組的資料，充分展現了事件驅動（Event-driven）架構的高效能特色。
*/

int main(int argc, char *argv[])
{
    int epfd, ready, fd, s, j, numOpenFds;;
    struct epoll_event ev;
    struct epoll_event evlist[MAX_EVENTS];
    char buf[MAX_BUF];

    if (argc < 2 || strcmp(argv[1], "--help") == 0)
        usageErr("%s file...\n", argv[0]);

    // 1. 初始化與建立 epoll 實體
    epfd = epoll_create(argc - 1);
    if (epfd == -1)
        errExit("epoll_create");

    /* Open each file on command line, and add it to the "interest
        list" for the epoll instance */

    // 2. 開啟檔案並註冊至「興趣清單」
    /*
        • 程式透過一個 for 迴圈開啟所有由命令列帶入的檔案。
        • 這裡特別注意 ev.data.fd = fd; 的動作，正對應了先前所述「data 欄位是找出與這個事件相關的檔案描述符之唯一機制」的標準作法。
        • 透過 epoll_ctl() 搭配 EPOLL_CTL_ADD 旗標，把這些 FD 一個個塞進核心的興趣清單中。
    */
    for (j = 1; j < argc; j++) {
        fd = open(argv[j], O_RDONLY);   // // 以唯讀方式開啟命令列指定的檔案/FIFO/管線

        if (fd == -1)
            errExit("open");
        printf("Opened \"%s\" on fd %d\n", argv[j], fd);

        ev.events = EPOLLIN;    // 設定只對「可讀取（Input）」事件感興趣   /* Only interested in input events */
        ev.data.fd = fd;        // 將 FD 編號存入 data 欄位，供日後識別

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) == -1)
            errExit("epoll_ctl");
    }

    // 3. 事件循環：等待與收割就緒事件
    /*
        • epoll_wait() 的第四個參數傳入 -1，代表永久阻塞。程式會在這裡停住、讓出 CPU 資源，直到被監控的檔案有資料進來（或發生錯誤）才會醒來。
        • 醒來後，ready 會回傳目前有多少個 FD 已經就緒，且最多一次塞滿 MAX_EVENTS（即 5 個）到 evlist 陣列中。
    */
    numOpenFds = argc - 1;
    while (numOpenFds > 0) {    // 只要還有被監控的 FD 沒關閉，就持續循環
        /* Fetch up to MAX_EVENTS items from the ready list */
        printf("About to epoll_wait()\n");
        ready = epoll_wait(epfd, evlist, MAX_EVENTS, -1);   // 傳入 -1，代表永久阻塞死等
        if (ready == -1) {
            if (errno == EINTR)
                continue;   // 如果是被系統訊號中斷，則重啟等待 /* Restart if interrupted by signal */
            else
                errExit("epoll_wait");
        }
        printf("Ready: %d\n", ready);

        /* Deal with returned list of events */

        // 4. 處理就緒事件清單
        //  • 解讀： 透過位元與運算（& EPOLLIN）確認是可讀事件，隨即呼叫 read() 讀取最多 1000 記憶體
        //          位元組的資料並印在螢幕上。
        for (j = 0; j < ready; j++) {
            printf(" fd=%d; events: %s%s%s\n", evlist[j].data.fd,
                (evlist[j].events & EPOLLIN) ? "EPOLLIN " : "",
                (evlist[j].events & EPOLLHUP) ? "EPOLLHUP " : "",
                (evlist[j].events & EPOLLERR) ? "EPOLLERR " : "");

            // 理分支 A：有資料可讀（EPOLLIN）
            if (evlist[j].events & EPOLLIN) {
                s = read(evlist[j].data.fd, buf, MAX_BUF);
                if (s == -1)
                    errExit("read");
                printf(" read %d bytes: %.*s\n", s, s, buf);
            
            // 處理分支 B：發生斷線或錯誤（EPOLLHUP 或 EPOLLERR）
            //  • 當對端掛斷（EPOLLHUP）或發生錯誤（EPOLLERR）時，且當前沒有 EPOLLIN 事件時，
            //    程式會執行 close() 關閉該 FD。
            //  • 隱藏的 Linux 特性：當你呼叫 close(fd) 時，核心會自動將該 fd 從該 epoll 實體
            //    的興趣清單中移除。因此程式不需要手動去呼叫 epoll_ctl(..., EPOLL_CTL_DEL, ...)。
            //  • 當 numOpenFds 歸零，代表所有檔案都讀完並關閉了，while 迴圈結束，程式優雅退出。
            } else if (evlist[j].events & (EPOLLHUP | EPOLLERR)) {
                /* If EPOLLIN and EPOLLHUP were both set, then there might
                    be more than MAX_BUF bytes to read. Therefore, we close
                    the file descriptor only if EPOLLIN was not set.
                    We'll read further bytes after the next epoll_wait(). */

                printf(" closing fd %d\n", evlist[j].data.fd);
                if (close(evlist[j].data.fd) == -1)
                    errExit("close");
                numOpenFds--;
            }
        }
    }
    printf("All file descriptors closed; bye\n");
    exit(EXIT_SUCCESS);
}
