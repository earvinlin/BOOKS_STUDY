/**
 * 
    假設使用者ID 值皆為X的行程執行了使用者ID為Y的 set-user-ID 程式，且.Y 值不為0，
    接著將行程憑證設定如下：
    real=X effective=y saved=Y
    (我們忽略檔案系統使用者ID，因為該ID 會跟隨有效使用者ID。)為執行如下操作，請分別列出對seluid()、
    sefeuid()、setreuidf()和 setresuid()呼叫。
    a) 暫停並恢復 set-user-ID 的身份(即將有效使用者ID 切換為真實使用者ID 的值，並接著切回 
       saved set-user-ID 的值)。
    b) 永久放棄 set-user-ID 的身份(即確保將有效使用者ID 和 saved sct-user-ID 設定為真實使用者ID)。
       (此習題還需要使用 getuid()和geteuid()函式，以檢素行程的真實使用者1D和有效使用者ID。)
    請注意上述列出的特定系統呼叫，其部分操作會無法進行。
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
