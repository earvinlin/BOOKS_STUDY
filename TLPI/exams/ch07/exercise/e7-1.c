/**
 * 修改列表7-1的程式(free_and_sbrk.c)，在每次執行malloc()之後，印出目前的
 * program break值。指定一個小的配置區塊來執行程式。這將能展示malloc()在每
 * 次呼明時不會用sbrk()調整program break，而是定期分配大塊的記憶體，並每次 
 * 傳回一小片記憶體給呼叫者。
 */
#include <stdlib.h>
//#include <string.h>
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
