/**
 * 
    假設在下列的各種情況中，行程使用者ID 的初始設定分別為real（真實）=1000、effcctive（有效）=0、
    saved（保存）=0、file-system（檔案系統）=0。當執行這些呼叫之後，使用者ID 的狀態如何？
    a) setuid(2000);
    b) setreuid(-1, 2000);
    c) seteuid(2000);
    d) setfsuid (2000);
    e) setresuid(-1, 2000, 3000);
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