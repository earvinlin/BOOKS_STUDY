/**
 * 
    使用 setgroups（）及函式庫函式從密碼檔、群組檔（參考8.4節）檢素資訊，以實作 initgroups（）。
    請記得，呼叫 setgroups（）的行程必須具有特權。
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
