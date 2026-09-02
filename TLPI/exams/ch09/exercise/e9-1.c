/**
 * 
    假設在下列的各種情況中，行程使用者ID 的初始設定分別為real（真實）=1000、effcctive（有效）=0、
    saved（保存）=0、file-system（檔案系統）=0。當執行這些呼叫之後，使用者ID 的狀態如何？
    a) setuid(2000);
    b) setreuid(-1, 2000);
    c) seteuid(2000);
    d) setfsuid (2000);
    e) setresuid(-1, 2000, 3000);

    ANS.
    這題考查的是 Linux/UNIX 中的特權行程（EUID = 0，即以 root 身份執行）呼叫各種修改 UID 的系統呼叫時，
    四個 UID 的變化規則。
    前提條件與核心原則
    • 初始狀態：
    • RUID (Real) = 1000（非 root 一般使用者）
    • EUID (Effective) = 0（特權狀態）
    • SUID (Saved) = 0
    • FSUID (File-system) = 0
    • 關鍵規則：因為當前的 EUID} = 0，該行程屬於特權行程（Superuser Process）。當特權行程變更 UID 時，
    規則與一般行程不同：
    各子題詳細解答與狀態變化    
    a) setuid(2000);
    • 動作與規則：當 EUID 為 0 的特權行程呼叫 setuid(val) 時，系統會將 RUID、EUID、SUID 全部強制修改為
      傳入的值，且 FSUID 也會跟著 EUID 改變。
    • 結果：
    • Real = 2000
    • Effective = 2000
    • Saved = 2000
    • File-system = 2000
    • 說明：此操作會徹底且永久地丟棄特權，無法再切換回 0。

    b) setreuid(-1, 2000);
    • 動作與規則：setreuid(ruid, euid) 用於同時設定 RUID 與 EUID。傳入 -1 表示保持不變。
    • 第一個參數 -1：RUID 保持為 1000。
    • 第二個參數 2000：EUID 設為 2000。
    • 附加效應：當 EUID 改變時，FSUID 預設跟著 EUID 變為 2000；此外，如果 EUID 被修改（或是指定了新的 
      RUID），特權行程的 SUID 會被設為新的 EUID（2000）。
    • 結果：
    • Real = 1000
    • Effective = 2000
    • Saved = 2000
    • File-system = 2000

    c) seteuid(2000);
    • 動作與規則：seteuid(euid) 只修改 EUID（以及連動的 FSUID），不會影響 RUID 與 SUID。
    • 結果：
    • Real = 1000
    • Effective = 2000
    • Saved = 0
    • File-system = 2000
    • 說明：因為 SUID 依然保留為 0，該行程雖然當下失去特權，但未來仍可以透過呼叫 seteuid(0) 重新恢復特權。

    d) setfsuid(2000);
    • 動作與規則：setfsuid(fsuid) 專門且僅修改 FSUID，用於控制檔案系統 access 權限，完全不影響 RUID、EUID 和 SUID。
    • 結果：
    • Real = 1000
    • Effective = 0
    • Saved = 0
    • File-system = 2000

    e) setresuid(-1, 2000, 3000);
    • 動作與規則：setresuid(ruid, euid, suid) 能精確且分別指定 RUID、EUID 與 SUID。-1 代表不變。
    • 第一個參數 -1：RUID 保持為 1000。
    • 第二個參數 2000：EUID 設為 2000。
    • 第三個參數 3000：SUID 設為 3000。
    • 連動效應：當 EUID 被修改時，FSUID 會自動同步改為新的 EUID (2000)。
    • 結果：
    • Real = 1000
    • Effective = 2000
    • Saved = 3000
    • File-system = 2000

    呼叫指令                        Real    Effective   Saved   File-system
                                  (RUID)  (EUID)      (SUID)  (FSUID)
    ======================================================================
    初始狀態                        1000	    0	        0	    0
    a) setuid(2000)                2000	    2000	    2000	2000
    b) setreuid(-1, 2000)	       1000	    2000	    2000	2000
    c) seteuid(2000)	           1000	    2000	       0	2000
    d) setfsuid(2000)	           1000	       0	       0	2000
    e) setresuid(-1, 2000, 3000)   1000	    2000	    3000	2000
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