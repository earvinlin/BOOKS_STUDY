/**
 * 
    擁有如下使用者ID 的行程具有特權嗎？請表達您的看法。
    real=0 effective=1000 saved=1000 file-system=1000

    ANS.
    這個題目考查的是 UNIX/Linux 作業系統中行程的使用者 ID（User ID, UID）權限管理機制。
    答案是：這個行程「當下不具有特權」，但它「隨時可以重新取得特權」。
    一、 各 UID 欄位的現狀解析
        在 Linux 中，UID 0 代表最高權限管理者（root），而 UID 1000 通常代表一般使用者：
        • Real UID (RUID) = 0 (root)：表示啟動這個行程的「真實使用者」是最高權限者 root。
        • Effective UID (EUID) = 1000 (一般使用者)：作業系統在當下做權限檢查（例如讀寫檔案、
          執行系統呼叫）時，看的就是 EUID。因為 EUID 為 1000，所以當下此行程只具備一般使用者的權
          限，不具備特權。
        • Saved Set-User-ID (SUID) = 1000：當行程切換權限時用來暫存 UID 的欄位。
        • File-system UID (FSUID) = 1000：Linux 專用，用來檢查檔案系統存取權限，目前也是一般
          使用者權限。
    二、 深入看法與分析
        1. 當前狀態：暫時放棄特權（Dropped Privileges）
            這是一種常見且良好的安全性設計範例（Principle of Least Privilege，最小權限原則）。
            這個行程最初是由 root (UID 0) 啟動的，但為了避免程式在執行過程中因漏洞遭到攻擊，它主
            動呼叫了系統呼叫（如 seteuid(1000)）將自己的 Effective UID 降低為 1000。因此在目
            前狀態下，它無法直接存取敏感的系統資源。
        2. 潛在能力：隨時可恢復特權
            因為它的 Real UID (RUID) 依然是 0，作業系統允許此行程隨時透過系統呼叫：
            seteuid(0); // 重新將 EUID 設回 0 
            把權限切換回 root。
    三、 結論
        • 答案摘要：若問**「當下（At the moment）」有沒有特權？沒有（因為 EUID = 1000）。但若問
          「本質上」**是不是特權行程？是的，因為它的 RUID = 0，擁有一把隨時能重啟 root 權限的鑰匙。
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{

    return 0;
}