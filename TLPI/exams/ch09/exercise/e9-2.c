/**
 * 
    擁有如下使用者ID 的行程具有特權嗎？請表達您的看法。
    real=0 effective=1000 saved=1000 file-system=1000
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