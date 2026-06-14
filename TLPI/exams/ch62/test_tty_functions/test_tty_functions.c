/*
    它實作了一個能夠安全處理系統訊號（Signals）的終端機模式切換程式。
    當一個程式把終端機改成 cbreak 或 raw 模式時，如果程式被使用者按 Ctrl-C 終止、或者按
    Ctrl-Z 暫停，終端機的設定並不會自動回復。這會導致使用者回到命令列（Shell）後，
    發現終端機畫面錯亂（例如按 Enter 沒反應、輸入的字看不見）。
    這段程式碼的核心目的，就是展示如何利用訊號處理器（Signal Handlers）在程式被終止或暫停
    時，完美還原終端機設定。
*/
#include <termios.h>
#include <signal.h>
#include <ctype.h>
#include "../tty_functions/tty_functions.h" /* Declarations of ttySetCbreak() and ttySetRaw() */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

// 1. 核心全域變數與通用終止處理
// userTermios：用來備份使用者最初進入程式前的終端機狀態（即標準加工模式 Cooked Mode）。
// handler()：通用的訊號處理函式。當程式收到終止訊號（如 SIGINT、SIGQUIT、SIGTERM）時，
// 這個處理器會攔截訊號，先把終端機還原成 userTermios，接著才乾淨地結束程式（_exit）。這
// 避免了終端機陷入永久殘廢的窘境。
static struct termios userTermios;
/* Terminal settings as defined by user */

/* General handler: restore tty settings and exit */
static void  handler(int sig) {
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &userTermios) == -1)
        errExit("tcsetattr");

    _exit(EXIT_SUCCESS);
}

// 2. 精妙的重頭戲：tstpHandler (處理 Ctrl-Z 暫停與恢復)
// 當使用者在 cbreak 模式下按下 Ctrl-Z，系統會發送 SIGTSTP 訊號來暫停程式。
// 這部分的處理非常精妙，其標準工作流程如下：
/* Handler for SIGTSTP */
static void tstpHandler(int sig) {
    struct termios ourTermios; /* To save our tty settings */
    sigset_t tstpMask, prevMask;
    struct sigaction sa;
    int savedErrno;
    savedErrno = errno; /* We might change 'errno' here */

    /* Save current terminal settings, restore terminal to
        state at time of program startup */

    // 1️⃣ 步驟一：備份目前狀態，還原原始終端機
    if (tcgetattr(STDIN_FILENO, &ourTermios) == -1) // 備份程式自己客製化的終端機狀態（如 cbreak）
        errExit("tcgetattr");

    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &userTermios) == -1) // 還原成使用者的原始終端機狀態，讓 Shell 接管時正常
        errExit("tcsetattr");

    /* Set the disposition of SIGTSTP to the default, raise the signal
        once more, and then unblock it so that we actually stop */

    // 2️⃣ 步驟二：自我引爆 SIGTSTP 真正進入暫停
    // 因為我們寫了自訂的 tstpHandler 攔截了 SIGTSTP，如果這時不把行為改回預設（SIG_DFL）並重新觸發（raise），
    // 程式就永遠不會真的暫停。
    if (signal(SIGTSTP, SIG_DFL) == SIG_ERR) /* 將 SIGTSTP 恢復為系統預設行為（即暫停） */
        errExit("signal");

    raise(SIGTSTP); /* 發送 SIGTSTP 給自己 */

    // 3️⃣ 步驟三：解開訊號遮罩（Unblock）
    // 核心在呼叫訊號處理器時，會自動將該訊號加入遮罩（Block）。如果不手動解開（Unblock），剛剛 raise 
    // 的暫停訊號就會被卡住。
    sigemptyset(&tstpMask);
    sigaddset(&tstpMask, SIGTSTP);

    if (sigprocmask(SIG_UNBLOCK, &tstpMask, &prevMask) == -1) /* 解鎖後，程式在此處真正暫停 */
        errExit("sigprocmask");
    
    // 📢 程式在此處暫停！ 當使用者在 Shell 輸入 fg 指令恢復程式時（系統發送 SIGCONT），
    // 程式會從這行程式碼之後繼續往下執行。

    /* Execution resumes here after SIGCONT */

    // 4️⃣ 步驟四：重回前景（Foreground），恢復自訂設定
    // 當程式醒來後，先把處理器重新掛好，接著把終端機再次切換回程式需要的 cbreak 狀態（當初備份的 ourTermios），
    // 讓迴圈能繼續無縫運作。
    if (sigprocmask(SIG_SETMASK, &prevMask, NULL) == -1)    /* 重新鎖定 SIGTSTP */
        errExit("sigprocmask"); /* Reblock SIGTSTP */

    sigemptyset(&sa.sa_mask); /* Reestablish handler */
    sa.sa_flags = SA_RESTART;
    sa.sa_handler = tstpHandler;

    if (sigaction(SIGTSTP, &sa, NULL) == -1)
        errExit("sigaction");

    /* The user may have changed the terminal settings while we were
        stopped; save the settings so we can restore them later */
    
    /* 重新掛載 tstpHandler 處理器 */

    if (tcgetattr(STDIN_FILENO, &userTermios) == -1)    /* 重新讀取，防止使用者在暫停期間改了終端機設定 */
        errExit("tcgetattr");
    
    /* Restore our terminal settings */
    
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &ourTermios) == -1)  /* 還原我們程式要用的 cbreak 狀態 */
        errExit("tcsetattr");
    
    errno = savedErrno;
}

int main(int argc, char *argv[])
{
    char ch;
    struct sigaction sa, prev;
    ssize_t n;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    if (argc > 1) { /* Use cbreak mode */
        if (ttySetCbreak(STDIN_FILENO, &userTermios) == -1)
            errExit("ttySetCbreak");
        
        /* Terminal special characters can generate signals in cbreak
         * mode. Catch them so that we can adjust the terminal mode.
         * We establish handlers only if the signals are not being ignored. */
        sa.sa_handler = handler;
        if (sigaction(SIGQUIT, NULL, &prev) == -1)
            errExit("sigaction");
        if (prev.sa_handler != SIG_IGN)
            if (sigaction(SIGQUIT, &sa, NULL) == -1)
                errExit("sigaction");
        if (sigaction(SIGINT, NULL, &prev) == -1)
            errExit("sigaction");

        /* 如果系統原先沒有忽略這些訊號，就掛載監聽處理器 */
        if (prev.sa_handler != SIG_IGN)
            if (sigaction(SIGINT, &sa, NULL) == -1)
                errExit("sigaction");
        sa.sa_handler = tstpHandler;
        if (sigaction(SIGTSTP, NULL, &prev) == -1)
            errExit("sigaction");
        if (prev.sa_handler != SIG_IGN)
            if (sigaction(SIGTSTP, &sa, NULL) == -1)
                errExit("sigaction");
    } else { /* Use raw mode */
        /* 注意：Raw 模式下不需處理 SIGINT/SIGTSTP，因為核心已經不解譯 Ctrl-C/Ctrl-Z 了 */
        if (ttySetRaw(STDIN_FILENO, &userTermios) == -1) // 沒參數：進 raw 模式
            errExit("ttySetRaw");
    }
    /*
        ．如果執行 ./program c，進入 cbreak 模式。此時鍵盤特殊鍵（Ctrl-C、Ctrl-Z）
          依然有效，因此必須掛載上述的訊號處理器來確保安全還原。
        ．如果執行 ./program，進入 raw 模式。此時 Ctrl-C / Ctrl-Z 只是普通字元，根
          本不會觸發訊號，所以不需要（也沒辦法）幫它們掛載處理器。
    */

    sa.sa_handler = handler;
    if (sigaction(SIGTERM, &sa, NULL) == -1)
        errExit("sigaction");

    setbuf(stdout, NULL); /* Disable stdout buffering */
    // 主迴圈：資料處理與 Echo 測試
    for (;;) { /* Read and echo stdin */
        n = read(STDIN_FILENO, &ch, 1);
        if (n == -1) {
            errMsg("read");
            break;
        }
        if (n == 0) /* Can occur after terminal disconnect */
            break;
        if (isalpha((unsigned char) ch)) /* Letters --> lowercase */
            putchar(tolower((unsigned char) ch));
        else if (ch == '\n' || ch == '\r')  // 換行符號正常印出
            putchar(ch);
        else if (iscntrl((unsigned char) ch)) // 控制字元轉換（如按 Ctrl-A 畫面印出 ^A
            printf("^%c", ch ^ 64); /* Echo Control-A as ^A, etc. */
        else
            putchar('*'); /* All other chars as '*' */ // 其他字元全部顯示為星號 '*'
        if (ch == 'q') /* Quit loop */ // 按下小寫 'q' 可以正常離開迴圈
            break;
    }
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &userTermios) == -1)
        errExit("tcsetattr");
    
    exit(EXIT_SUCCESS);
}
