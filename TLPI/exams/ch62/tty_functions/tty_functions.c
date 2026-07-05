#include <termios.h>
#include <unistd.h>
/* Declares functions defined here */
/* 
    Place terminal referred to by 'fd' in cbreak mode (noncanonical mode
    with echoing turned off). This function assumes that the terminal is
    currently in cooked mode (i.e., we shouldn't call it if the terminal
    is currently in raw mode, since it does not undo all of the changes
    made by the ttySetRaw() function below). Return 0 on success, or -1
    on error. If 'prevTermios' is non-NULL, then use the buffer to which
    it points to return the previous terminal settings. 
*/
#include "tty_functions.h" 


// 切換至 cbreak 模式
int ttySetCbreak(int fd, struct termios *prevTermios) {
    struct termios t;
    
    // 共通的安全防護：備份機制
    // tcgetattr(fd, &t)：先取出目前終端機的設定，存入臨時變數 t。
    // *prevTermios = t：如果呼叫者有傳入用來備份的指標（非 NULL），就
    // 把目前的設定複製一份存進去。這樣一來，外層程式在結束前，就能用這個
    // 備份將終端機還原，避免前面提到的 EOF（Ctrl-D）被永久覆寫的 Bug。
    if (tcgetattr(fd, &t) == -1)
        return -1;
    if (prevTermios != NULL)
        *prevTermios = t;

    /*
        位元操作拆解：
        t.c_lflag &= ~(ICANON | ECHO);
            ~ICANON：關閉規範模式（進入非規範模式）。自此，終端機不再需要等使用者按 Enter，任何按鍵都會立刻送出。
            ~ECHO：關閉回顯。使用者敲擊鍵盤時，螢幕上不會顯示該字元（這在做全螢幕選單或遊戲時很常用，避免畫面變亂）。
        t.c_lflag |= ISIG;
            | ISIG：強制開啟訊號處理。確保 Ctrl-C (INTR)、Ctrl-Z (SUSP) 這些快捷鍵按下去時，核心依然會發送訊號來中斷程式。
        t.c_iflag &= ~ICRNL;
            ~ICRNL：關閉將 CR（Carriage Return, \r）轉換為 NL（New Line, \n）的功能。這樣程式就能精準抓到使用者到底按的是 Enter 鍵（通常是 CR）還是真正的換行。
        t.c_cc[VMIN] = 1; 且 t.c_cc[VTIME] = 0;
            VMIN = 1 代表「只要有 1 個位元組進來就立刻返回」。
            VTIME = 0 代表「無限期阻塞直到有資料進來」。
            兩者結合：程式會卡在 read() 行，直到使用者敲擊任意一個按鍵，read() 就會立刻帶著 1 個字元返回。    
    */
    t.c_lflag &= ~(ICANON | ECHO);
    t.c_lflag |= ISIG;
    t.c_iflag &= ~ICRNL;
    t.c_cc[VMIN] = 1; /* Character-at-a-time input */
    t.c_cc[VTIME] = 0; /* with blocking */

    if (tcsetattr(fd, TCSAFLUSH, &t) == -1)
        return -1;
    
    return 0;
}

/* 
    Place terminal referred to by 'fd' in raw mode (noncanonical mode
    with all input and output processing disabled). Return 0 on success,
    or -1 on error. If 'prevTermios' is non-NULL, then use the buffer to
    which it points to return the previous terminal settings. 
*/
// 切換至原始模式（Raw Mode）
int ttySetRaw(int fd, struct termios *prevTermios) {
    struct termios t;

    if (tcgetattr(fd, &t) == -1)
        return -1;
    if (prevTermios != NULL)
        *prevTermios = t;

    /*
    t.c_lflag &= ~(ICANON | ISIG | IEXTEN | ECHO);
        ．除了關閉 ICANON 和 ECHO，它還關閉了 ISIG（這意味著 Ctrl-C、Ctrl-Z 徹底失效，
          核心不再幫忙處理訊號）以及 IEXTEN（關閉擴充輸入處理）。
    t.c_iflag &= ~(BRKINT | ICRNL | IGNBRK | IGNCR | INLCR | INPCK | ISTRIP | IXON | PARMRK);
        ．這一大串清單關閉了幾乎所有的輸入加工。
        ．例如：不處理 Break 鍵（~BRKINT/~IGNBRK）、不轉換或忽略換行符號（~IGNCR/~INLCR）、
          不進行奇偶校驗與位元剝離（~INPCK/~ISTRIP）。
        ．特別注意 ~IXON：關閉了軟體串流控制（XON/XOFF）。平時我們按 Ctrl-S 會讓終端機暫停輸出、
          Ctrl-Q 恢復，在 Raw 模式下，這兩個鍵也會被當作普通二進位資料讀進來，不會暫停終端機。
    t.c_oflag &= ~OPOST;
        ．~OPOST：關閉所有輸出加工（Output Post-Processing）。在普通模式下，程式輸出 \n，驅動程
          式會自動幫你補成 \r\n 才能正確換行；一旦關閉 OPOST，輸出 \n 就真的只是純換行（游標垂直
          向下，不會回到行首），所有輸出格式控制都必須由應用程式自己肉搏處理。
        ．t.c_cc[VMIN] = 1; 且 t.c_cc[VTIME] = 0;
            同樣保持「有 1 個字元就立刻返回，沒字元就阻塞」的即時讀取模式。
    */
    
    t.c_lflag &= ~(ICANON | ISIG | IEXTEN | ECHO);
    
    /* Noncanonical mode, disable signals, extended
        input processing, and echoing */
    t.c_iflag &= ~(BRKINT | ICRNL | IGNBRK | IGNCR | INLCR |
        INPCK | ISTRIP | IXON | PARMRK);

    /* Disable special handling of CR, NL, and BREAK.
        No 8th-bit stripping or parity error handling.
        Disable START/STOP output flow control. */
    t.c_oflag &= ~OPOST; /* Disable all output processing */
    t.c_cc[VMIN] = 1; /* Character-at-a-time input */
    t.c_cc[VTIME] = 0; /* with blocking */

    // 關鍵的寫入生效
    // 參數是 TCSAFLUSH，這是一個非常專業且安全的細節：
    // 它的意思是：等到目前輸出緩衝區的資料全部傳送完畢後，才改變設定。
    // 更重要的是，它會將目前輸入緩衝區中「還沒被程式 read 讀取」的舊資料全部丟棄（Flush）。
    // 為什麼要這樣做？ 因為在切換模式前，使用者可能胡亂敲了一些字（當時在 Cooked 模式下），
    // 如果不清空，一轉換到 cbreak/raw 模式，程式就會立刻讀到這些殘留的錯誤舊資料。
    if (tcsetattr(fd, TCSAFLUSH, &t) == -1)
        return -1;

    return 0;
}
/*
    這是一份非常標準且教科書等級的系統程式碼（通常出自經典著作 The Linux Programming Interface ）。
    如果你要寫終端機遊戲（如貪食蛇）、vim 編輯器、less 閱讀器：應該呼叫 ttySetCbreak()。
    如果你要寫序列埠（UART）二進位通訊、遠端連線工具（如 SSH/Telnet 伺服器底層）、檔案傳輸工具：
    則必須呼叫 ttySetRaw()。
*/
