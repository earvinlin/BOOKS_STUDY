#include <time.h>
#include <poll.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    int numPipes, j, ready, randPipe, numWrites;
    int (*pfds)[2]; /* File descriptors for all pipes */
    struct pollfd *pollFd;

    if (argc < 2 || strcmp(argv[1], "--help") == 0)
        usageErr("%s num-pipes [num-writes]\n", argv[0]);

    /* Allocate the arrays that we use. The arrays are sized according
        to the number of pipes specified on command line */

    // 動態記憶體配置：從命令列參數（argv[1]）讀取使用者想要建立幾條管線（例如輸入 1000）。
    //               著用 calloc 動態挖出兩塊記憶體
    // pfds：用來存放 1000 條管線的檔案描述符（每條管線包含 [0] 讀取端與 [1] 寫入端，共 
    //       2000 個 fd）。
    // pollFd：這就是 poll() 的報名表清單！ 總共有 1000 個 struct pollfd 結構體。
    numPipes = getInt(argv[1], GN_GT_0, "num-pipes");
    pfds = calloc(numPipes, sizeof(int [2]));
    if (pfds == NULL)
        errExit("malloc");

    pollFd = calloc(numPipes, sizeof(struct pollfd));
    if (pollFd == NULL)
        errExit("malloc");

    /* Create the number of pipes specified on command line */

    // 建立一對一管道：利用迴圈呼叫 pipe()。此時作業系統會在核心建立 1000 條虛擬管道。
    // pfds[j][0] ➔ 這條管道的出水口（讀取端）。
    // pfds[j][1] ➔ 這條管道的進水口（寫入端）。
    for (j = 0; j < numPipes; j++)
        if (pipe(pfds[j]) == -1)
            errExit("pipe %d", j);

    /* Perform specified number of writes to random pipes */

    numWrites = (argc > 2) ? getInt(argv[2], GN_GT_0, "num-writes") : 1;    // 要灌水幾次
    srandom((int) time(NULL));  // 隨機數種子
    for (j = 0; j < numWrites; j++) {
        randPipe = random() % numPipes; // 隨機挑選一條幸運管線
        printf("Writing to fd: %3d (read fd: %3d)\n", pfds[randPipe][1], pfds[randPipe][0]);

        // 往進水口寫入 1 個位元組的字母 "a"
        if (write(pfds[randPipe][1], "a", 1) == -1)
            errExit("write %d", pfds[randPipe][1]);
    }

    /* Build the file descriptor list to be supplied to poll(). This list
        is set to contain the file descriptors for the read ends of all of
        the pipes. */

    // /* 填寫報名表清單 */
    for (j = 0; j < numPipes; j++) {
        pollFd[j].fd = pfds[j][0];  // 我要監控第 j 條管線的「讀取端」
        pollFd[j].events = POLLIN;  // 我對「有沒有資料流進來 (POLLIN)」這件事感興趣
    }

    /* 呼叫大魔王 */
    /*
    核心監控 poll()：
    • 第一個參數 pollFd：把整張 1000 條管線的報名表交給核心。
    • 第二個參數 numPipes：告訴核心這張表有 1000 列。
    • 第三個參數 -1：代表「無限期阻塞等待」。直到這 1000 條管線中，至少有一條管線有人灌水進去，
                   poll() 才會被喚醒並返回。（註：原始代碼中的註解寫 / * Nonblocking * / 
                   其實是寫錯了或筆誤，帶 -1 絕對是阻塞式死等）。
    當 poll() 成功返回時，返回值 ready 代表**「總共有幾條管線處於就緒狀態」**（如果剛才順利灌水 
    3 次，這裡就會回傳 3）。
    */
    ready = poll(pollFd, numPipes, -1); /* Nonblocking */
    if (ready == -1)
        errExit("poll");

    printf("poll() returned: %d\n", ready);

    /* Check which pipes have data available for reading */

    /*
        作業系統醒來後，會把有資料的清單列中的 .revents（回報事件）欄位填上 POLLIN。
        程式必須用一個 for 迴圈，從第 0 條一路看到第 999 條，用位元與運算（& POLLIN）
        去檢查：「是你嗎？還是你？」。最終，它會精準地把剛才被隨機選中灌水的那 3 條管線編號
        與 fd 給列印出來。
    */
    for (j = 0; j < numPipes; j++)
        if (pollFd[j].revents & POLLIN)
            printf("Readable: %d %3d\n", j, pollFd[j].fd);

    exit(EXIT_SUCCESS);
}

/*
    這段代碼極其優雅地展示了 poll() 的標準運作模型。它突破了 select() 的 1024 限制，可以開到極
    大的管道數。
    然而，你也可以從最後一步看出 poll() 的歷史局限性：即使只有 3 條管道有資料，程式依然必須寫一個 
    for (j = 0; j < numPipes; j++) 去完整跑完 1000 次迴圈，才能把這 3 個幸運兒找出來。當管線
    高達十萬(N=100,000)時，這種盲目的全量大搜查就會開始拖慢速度，這也是為什麼 Linux 後來又誕生了
    更終極的 epoll（直接把就緒的 3 個 fd 打包成一個小陣列丟回來，免去全量迴圈）的原因！
*/
