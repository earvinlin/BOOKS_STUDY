#include <signal.h>
#include <ctype.h>
#include <fcntl.h>
#include <termios.h>
#include "../../../tlpi-book/tty/tty_functions.h" /* Declaration of ttySetCbreak() */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

// volatile sig_atomic_t：這種類型確保變數在信號處理器與主程式之間共享時，
// 讀寫操作是原子性（Atomic）且不會被編譯器優化到暫存器中，保證記憶體同步。
static volatile sig_atomic_t gotSigio = 0;

/* Set nonzero on receipt of SIGIO */

static void sigioHandler(int sig) {
    gotSigio = 1;
}

int main(int argc, char *argv[])
{
    int flags, j, cnt;
    struct termios origTermios;
    char ch;
    struct sigaction sa;
    Boolean done;

    /* Establish handler for "I/O possible" signal */

    sigemptyset(&sa.sa_mask);

    // 註冊信號處理器 
    // 告訴系統：當 SIGIO 發生時，請執行 sigioHandler。
    sa.sa_flags = SA_RESTART;
    sa.sa_handler = sigioHandler;
    if (sigaction(SIGIO, &sa, NULL) == -1)
        errExit("sigaction");

    /* Set owner process that is to receive "I/O possible" signal */

    // 指定信號接收者（F_SETOWN）
    // 這一步非常關鍵。作業系統必須知道當 STDIN_FILENO（標準輸入）有資料進來時，
    // 要把 SIGIO 信號送給哪一個進程。這裡傳入 getpid() 表示送給當前程式。
    if (fcntl(STDIN_FILENO, F_SETOWN, getpid()) == -1)
        errExit("fcntl(F_SETOWN)");

    /* Enable "I/O possible" signaling and make I/O nonblocking
        for file descriptor */
        
    // 啟用非同步 I/O（O_ASYNC）和非阻塞 I/O（O_NONBLOCK）
    // O_ASYNC：正式啟動信號驅動式 I/O。當檔案描述符（FD）可以進行 I/O 操作時，
    //          核心就會產生 SIGIO 信號。
    // O_NONBLOCK：將 I/O 設為非阻塞。這是信號驅動 I/O 的標配，因為信號只告訴你
    //            「有資料可讀」，但沒說有多少。我們必須用非阻塞模式一直讀到回傳 
    //             EAGAIN 為止，否則程式會卡在 read 系統調用裡。
    flags = fcntl(STDIN_FILENO, F_GETFL);
    if (fcntl(STDIN_FILENO, F_SETFL, flags | O_ASYNC | O_NONBLOCK) == -1)
        errExit("fcntl(F_SETFL)");

    /* Place terminal in cbreak mode */

    // 終端機模式調整 (ttySetCbreak)
    // 預設情況下，終端機是「標準模式（Canonical Mode）」，必須按下 Enter 鍵才
    // 會把資料送給程式。
    // ttySetCbreak（TLPI 自訂函數）將終端機改為 Cbreak 模式（非標準模式的一
    // 種）。這使得使用者每敲一個按鍵，核心就立刻收到，不需要等 Enter，這才能觸發
    // 即時的 SIGIO。
    if (ttySetCbreak(STDIN_FILENO, &origTermios) == -1)
        errExit("ttySetCbreak");

    // 主迴圈運作邏輯（Main Loop）
    // 1. 模擬背景工作：外層的 cnt++ 和內層空轉的 j 迴圈，用來模擬程式正在執行某種
    //    繁重的背景計算。
    // 2. 主動檢查（Polling-like inside signal）：每次空轉完，程式會檢查 
    //    gotSigio 旗標。
    // 3. 非阻塞讀取：如果 gotSigio 為 1，代表在空轉期間使用者敲了鍵盤。程式進入 
    //    while 迴圈一次性讀完所有快取區裡的字元，直到 read 返回小於等於 0 的值
    //    (通常是因為非阻塞而返回 -1 且 errno 為 EAGAIN）。
    // 4. 結束條件：如果輸入的字元是 #，done 變為 TRUE，退出迴圈。
    for (done = FALSE, cnt = 0; !done ; cnt++) {
        for (j = 0; j < 100000000; j++)
            continue; /* Slow main loop down a little */

            if (gotSigio) { /* Is input available? */
                /* Read all available input until error (probably EAGAIN)
                    or EOF (not actually possible in cbreak mode) or a
                    hash (#) character is read */
                while (read(STDIN_FILENO, &ch, 1) > 0 && !done) {
                    printf("cnt=%d; read %c\n", cnt, ch);
                    done = ch == '#';
                }
                gotSigio = 0;
            }
        }

    /* Restore original terminal settings */

    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &origTermios) == -1)
        errExit("tcsetattr");

    exit(EXIT_SUCCESS);
}