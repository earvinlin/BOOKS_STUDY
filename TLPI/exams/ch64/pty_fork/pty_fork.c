#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include "../../../tlpi-book/mylib/pty_master_open.h"
#include "../../../tlpi-book/mylib/pty_fork.h" /* Declares ptyFork() */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define MAX_SNAME 1000

pid_t ptyFork(int *masterFd, char *slaveName, size_t snLen,
    const struct termios *slaveTermios, const struct winsize *slaveWS) {
    int mfd, slaveFd, savedErrno;
    pid_t childPid;
    char slname[MAX_SNAME];

    // 1. 準備階段：打開 Master 端並取得 Slave 名稱
    /*
        • 核心呼叫 ptyMasterOpen。這個自訂函式在底層會打開如 /dev/ptmx 的裝置，讓核心
          配置一對 PTY。
        • 它會傳回 mfd（Master 的檔案描述符），並將配對到的 Slave 裝置路徑字串（例如 
          /dev/pts/3）寫入 slname 變數中。
    */
    mfd = ptyMasterOpen(slname, MAX_SNAME);
    if (mfd == -1)
        return -1;

    if (slaveName != NULL) { /* Return slave name to caller */
        if (strlen(slname) < snLen) {
            strncpy(slaveName, slname, snLen);
        } else { /* 'slaveName' was too small */
            close(mfd);
            errno = EOVERFLOW;
            return -1;
        }
    }

    // 2. 分身階段：呼叫 fork()
    // 這裡系統分裂成兩個行程 : Child, Parent
    childPid = fork();
    if (childPid == -1) { /* fork() failed */
        savedErrno = errno; /* close() might change 'errno' */
        close(mfd); /* Don't leak file descriptors */
        errno = savedErrno;
        return -1;
    }

    // 家長端非常簡單，它把 Master 的檔案描述符（`mfd`）記下來（用來控制、讀寫
    // 子行程的輸入輸出），然後就直接返回。
    // 這對應了圖中左邊的 `driver program`。 (eng.version p.1377)
    if (childPid != 0) { /* Parent */
        *masterFd = mfd; /* Only parent gets master fd */
        return childPid; /* Like parent of fork() */
    }

    /* Child falls through to here (翻譯：子進程(程序)直接進入此處) */
    // 3. 子行程的蛻變（Child 核心重頭戲）
    //    當 childPid == 0 時，子行程開始執行一系列關鍵的系統呼叫，將自己徹底偽
    //    裝成一個終端機程式。

    // 【步驟 A：脫離舊 session，建立新 Session】
    // 原因： 子行程必須脫離父行程原本的控制終端機（Controlling Terminal）與程序
    // 群組。建立新的 Session 後，這個子行程會成為新群組的 Leader，並且此時暫時沒
    // 有任何控制終端機。
    if (setsid() == -1) /* Start a new session */
        err_exit("ptyFork:setsid");

    close(mfd); /* Not needed in child */

    // 【步驟 B：關閉 Master，打開 Slave】
    // 子行程呼叫 open(slname, ...) 去打開剛才配對出來的 Slave 裝置（如 /dev/pts/3）。
    // 在 System V 系統（如 Linux）中，一個沒有控制終端機的 Session Leader 只要第一次打
    // 開一個終端機裝置，該裝置就會自動自動成為它的控制終端機（Controlling TTY）。這步實現
    // 了圖中那條右側的弧線虛線。
    slaveFd = open(slname, O_RDWR); /* Becomes controlling tty */
    if (slaveFd == -1)
        err_exit("ptyFork:open-slave");

#ifdef TIOCSCTTY /* Acquire controlling tty on BSD */
    if (ioctl(slaveFd, TIOCSCTTY, 0) == -1)
        err_exit("ptyFork:ioctl-TIOCSCTTY");
#endif

    // 【步驟 C：複製終端機屬性與視窗大小】
    // 如果呼叫者有傳入設定，這裡會設定 Slave 的 termios
    // （如字元輸入模式、回顯 Echo 是否開啟）和 winsize（視窗的長寬、行列數）。
    if (slaveTermios != NULL) /* Set slave tty attributes */
        if (tcsetattr(slaveFd, TCSANOW, slaveTermios) == -1)
            err_exit("ptyFork:tcsetattr");

    if (slaveWS != NULL) /* Set slave tty window size */
        if (ioctl(slaveFd, TIOCSWINSZ, slaveWS) == -1)
            err_exit("ptyFork:ioctl-TIOCSWINSZ");

    /* Duplicate pty slave to be child's stdin, stdout, and stderr */

    // 【步驟 D：移花接木（dup2 標準輸入輸出重導向）】
    // 將子行程的 0 (stdin)、1 (stdout)、2 (stderr) 全部複製並替換為 slaveFd。
    // 經過這三行，子行程未來所有 printf（輸出）或 scanf（輸入），表面上是在對螢幕鍵
    // 盤操作，實際上全部都會流進這個 pty slave 裝置。
    if (dup2(slaveFd, STDIN_FILENO) != STDIN_FILENO)
        err_exit("ptyFork:dup2-STDIN_FILENO");
    if (dup2(slaveFd, STDOUT_FILENO) != STDOUT_FILENO)
        err_exit("ptyFork:dup2-STDOUT_FILENO");
    if (dup2(slaveFd, STDERR_FILENO) != STDERR_FILENO)
        err_exit("ptyFork:dup2-STDERR_FILENO");

    // 【步驟 E：清理與返回】
    /*
        • 原本打開的 slaveFd 已經被複製到 0, 1, 2 了，原本的數字就沒用了，為了省資源
          把它關掉。
        • 最後 return 0，代表子行程成功初始化 PTY 環境，準備好讓呼叫者在後面接續執行
          exec()（例如執行 /bin/bash）。    
    */
    if (slaveFd > STDERR_FILENO) /* Safety check */
        close(slaveFd); /* No longer need this fd */

    return 0; /* Like child of fork() */
}
