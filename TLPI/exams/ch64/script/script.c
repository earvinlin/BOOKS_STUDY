#include <sys/stat.h>
#include <fcntl.h>
#include <libgen.h>
#include <termios.h>
#include <sys/select.h>
#include "pty_fork.h" /* Declaration of ptyFork() */
#include "tty_functions.h" /* Declaration of ttySetRaw() */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define BUF_SIZE 256
#define MAX_SNAME 1000

struct termios ttyOrig;

/* Reset terminal mode on program exit */
static void ttyReset(void) {
    if (tcsetattr(STDIN_FILENO, TCSANOW, &ttyOrig) == -1)
        errExit("tcsetattr");
}

int main(int argc, char *argv[])
{
    char slaveName[MAX_SNAME];
    char *shell;
    int masterFd, scriptFd;
    struct winsize ws;
    fd_set inFds;
    char buf[BUF_SIZE];
    ssize_t numRead;
    pid_t childPid;

    // 1. 紀錄並獲取實體終端機的屬性
    /*
        在分裂行程之前，主程式必須先向作業系統查詢目前使用者實體終端機（STDIN_FILENO）的配置，
        包含：
        • ttyOrig（終端機屬性）： 紀錄了目前的快捷鍵設定、Echo 回顯是否開啟等。
        • ws（視窗大小）： 紀錄了目前終端機視窗的行列數（Rows & Columns）。這是為了稍後把這套
          設定原封不動地下傳給虛擬終端機，讓裡面的 Shell 感覺就像在外面一樣。
    */
    if (tcgetattr(STDIN_FILENO, &ttyOrig) == -1)
        errExit("tcgetattr");
    if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) < 0)
        errExit("ioctl-TIOCGWINSZ");

    // 2. 行程分裂：子行程執行 Shell
    /*
        • 呼叫我們先前看過的 ptyFork()。此時子行程已經被丟進 pty slave 的環境中，並且標準輸
          入輸出都已重新導向。
        • 子行程獲取系統環境變數中預設的 SHELL（例如 /bin/bash），並呼叫 execlp() 啟動它。
        • 對照架構圖： 這一步建立了圖右側的 shell (child) 與 pty slave。    
    */
    childPid = ptyFork(&masterFd, slaveName, MAX_SNAME, &ttyOrig, &ws);
    if (childPid == -1)
        errExit("ptyFork");
    if (childPid == 0) { /* Child: execute a shell on pty slave */
        shell = getenv("SHELL");
        if (shell == NULL || *shell == '\0')
            shell = "/bin/sh";

        execlp(shell, shell, (char *) NULL);
        errExit("execlp"); /* If we get here, something went wrong */
    }

    /* Parent: relay data between terminal and pty master */

    // 3. 家長行程的魔術（一）：打開紀錄檔與 Raw 模式
    scriptFd = open((argc > 1) ? argv[1] : "typescript",
        O_WRONLY | O_CREAT | O_TRUNC,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP |
        S_IROTH | S_IWOTH);

    if (scriptFd == -1)
        errExit("open typescript");
    
    // 打開用來儲存紀錄的檔案。如果使用者沒有指定檔名，預設就是常聽到的 "typescript"。
    /*
        • 極重要動作： 呼叫 ttySetRaw() 將最外層的實體終端機設定為 Raw 模式（原始模式）。
        • 為什麼要這樣做？ 在 Raw 模式下，內核會停用實體終端機的「回顯（Echo）」與「規範模式
         （行緩衝）」。當使用者按下任何按鍵（甚至是 Ctrl-C、Backspace），實體終端機都不會在
          螢幕上顯示任何字，而是立刻把這 1 個位元組原封不動地送交給 script 行程。
        • 同時註冊 atexit(ttyReset)，確保當程式退出時，會自動把使用者的實體終端機還原回原本
          的正常狀態，否則使用者的視窗會壞掉（打字沒反應、看不到字）。
    */
    ttySetRaw(STDIN_FILENO, &ttyOrig);

    if (atexit(ttyReset) != 0)
        errExit("atexit");

    // 4. 家長行程的魔術（二）：select() 
    // 這是整段程式碼的靈魂，它實現了架構圖中所有密密麻麻的實線箭頭。因為外層鍵盤什麼時候有輸入、內
    // 層 Shell 什麼時候有輸出，時間點完全無法預測，所以必須使用 select() 進行異步的多工監聽。
    for (;;) {
        FD_ZERO(&inFds);
        FD_SET(STDIN_FILENO, &inFds);
        FD_SET(masterFd, &inFds);

        // 當 select() 解除阻塞，代表其中一端有動靜，程式開始進行中繼搬運：
        if (select(masterFd + 1, &inFds, NULL, NULL, NULL) == -1)
            errExit("select");

        // ➡️ 狀況 A：使用者敲擊鍵盤（STDIN_FILENO 有資料
        /*
            • script 從實體鍵盤讀到字元，然後呼叫 write(masterFd, ...) 把它塞進 
              PTY Master，資料會流向內層的 Shell。
            • 注意： 這裡程式碼沒有把使用者的輸入寫入 scriptFd！因為只要內層的 
              pty slave 留著 Echo 功能，Shell 收到字元後會自動把字元「彈回來」
              （回顯）。所以輸入的字元會在下面狀況 B 被一併紀錄，這樣可以完美避免重複
              寫入。
        */
        if (FD_ISSET(STDIN_FILENO, &inFds)) { /* stdin --> pty */
            numRead = read(STDIN_FILENO, buf, BUF_SIZE);

            if (numRead <= 0)
                exit(EXIT_SUCCESS);

            if (write(masterFd, buf, numRead) != numRead)
                fatal("partial/failed write (masterFd)");
        }

        // ⬅️ 狀況 B：內層 Shell 有畫面輸出（masterFd 有資料）
        /*
            • 當內層 Shell 輸出資料（不論是執行的結果如 ls 的輸出，或是剛才鍵盤彈回來的
              Echo 字元），script 從 masterFd 撈出資料。
            • 接著執行一魚兩吃：一份送到外層螢幕（STDOUT_FILENO），一份送到歷史檔案
             （scriptFd）。
        */
        if (FD_ISSET(masterFd, &inFds)) { /* pty --> stdout+file */
            numRead = read(masterFd, buf, BUF_SIZE);

            if (numRead <= 0)
                exit(EXIT_SUCCESS);

            if (write(STDOUT_FILENO, buf, numRead) != numRead)
                fatal("partial/failed write (STDOUT_FILENO)");

            if (write(scriptFd, buf, numRead) != numRead)
                fatal("partial/failed write (scriptFd)");
        }
    }
}
