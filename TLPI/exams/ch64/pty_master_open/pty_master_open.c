/*
    define _XOPEN_SOURCE 600：這是一個功能測試巨集（Feature Test Macro）。
    設定為 600 代表告訴編譯器「我要使用符合 POSIX.1-2001 / SUSv3 標準的函式」。
    像是 posix_openpt() 這種標準函式，必須要有這個宣告才能正常編譯。
*/ 
#define _XOPEN_SOURCE 600
#include <stdlib.h>
#include <fcntl.h>
#include "pty_master_open.h" /* Declares ptyMasterOpen() */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int ptyMasterOpen(char *slaveName, size_t snLen) {
    int masterFd, savedErrno;
    char *p;

    // 2. 核心步驟：開啟 Master 裝置
    /*
        posix_openpt()： 這是標準的 PTY 建立方式。系統會自動在幕後幫你找到一個沒人使用的 Master 裝置並將它開啟。
        O_RDWR： 以可讀可寫模式開啟。
        O_NOCTTY： 非常重要！防止這個剛開啟的 PTY 裝置自動變成目前行程的「控制終端機（Controlling Terminal）」。此時我們只想管理它，還不想讓它成為主控台。
    */
    masterFd = posix_openpt(O_RDWR | O_NOCTTY); /* Open pty master */

    if (masterFd == -1)
        return -1;

    // 4. 初始化權限與解鎖
    // grantpt(masterFd)： 呼叫我們在第一張圖學到的機制（幕後執行 pt_chown），將 Slave 裝置的擁有者改成目前使用者、群組設為 tty。
    if (grantpt(masterFd) == -1) { /* Grant access to slave pty */
        // 3. 安全防禦：保存 errno 的手法
        /* 
            為什麼要這麼麻煩？ 這是高階系統程式設計的嚴謹細節。當 grantpt() 或 unlockpt() 失敗時，
            系統會設定錯誤碼（errno），告訴上層呼叫者失敗原因。
            然而，在結束函式前，我們必須呼叫 close(masterFd) 來釋放資源。close() 系統呼叫本身也有
            可能成功或失敗，進而覆蓋（修改）掉原本由 grantpt 引發的真正錯誤碼。
            因此，先把原本的錯誤存進 savedErrno，關閉檔案後再將它还原給 errno，確保外部呼叫者能拿到
            「最原始、正確的錯誤原因」。
        */              
        savedErrno = errno;
        close(masterFd); /* Might change 'errno' */
        errno = savedErrno;
    
        return -1;
    }

    // 4. 初始化權限與解鎖
    // unlockpt(masterFd)： 呼叫第三張圖學到的機制，解開核心對 Slave 裝置的「內鎖」。這步做完，其他行程才可以合法 open 這個 Slave 裝置。
    if (unlockpt(masterFd) == -1) { /* Unlock slave pty */
        savedErrno = errno;
        close(masterFd); /* Might change 'errno' */
        errno = savedErrno;
        
        return -1;
    }

    // 5. 取得 Slave 名稱與緩衝區安全檢查
    /*
        ．ptsname(masterFd)： 傳入 Master 的檔案描述符，核心會回傳該 Slave 裝置在系統中的路徑字串指標（例如回傳 /dev/pts/5）。
        ．安全檢查 if (strlen(p) < snLen)： 這呼應了第四張圖的參數設計。函式檢查核心回傳的路徑字串長度是否小於呼叫者提供的緩衝區大小（snLen）。
            - 空間足夠： 安全地使用 strncpy 將路徑複製到 slaveName。
            - 空間不足： 拒絕寫入，並手動將錯誤碼設定為 EOVERFLOW（值太大家，導致緩衝區溢位），關閉 PTY 並回傳失敗。    
    */
    p = ptsname(masterFd); /* Get slave pty name */
    if (p == NULL) {
        savedErrno = errno;
        close(masterFd); /* Might change 'errno' */
        errno = savedErrno;
        
        return -1;
    }

    if (strlen(p) < snLen) {
        strncpy(slaveName, p, snLen);
    } else { /* Return an error if buffer too small */
        close(masterFd);
        errno = EOVERFLOW;
        
        return -1;
    }
    // 6. 成功回傳
    return masterFd;
}
