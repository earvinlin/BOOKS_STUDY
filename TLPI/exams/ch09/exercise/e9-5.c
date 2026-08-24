/**
 * 
    以一個執行 set-user-ID-root 程式的行程，重複上個習題，其行程憑證之初始設定如下：
    real=X effective=0 saved=0
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
