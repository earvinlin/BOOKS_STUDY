/**
 * 設計一個程式，觀察當使用longjmp（）函式跳轉到一個已經返回的函式時會發生什麼事？
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#include <stdio.h>
#include <setjmp.h>

jmp_buf env;

void do_something() {
    printf("進入 do_something 函式\n");
    if (setjmp(env) == 0) {
        printf("setjmp 設置完成，準備離開 do_something 函式\n");
    } else {
        // 當 longjmp 跳回這裡時，do_something 的 Stack Frame 早就已經失效了！
        printf("成功跳回 do_something！(但此處 Stack 已經壞掉)\n");
    }
}

int main(int argc, char *argv[])
{
do_something(); // 呼叫函式，會在內部呼叫 setjmp 後 return 結束

    printf("返回 main 函式，準備呼叫 longjmp 跳回已結束的函式...\n");
    
    // 試圖跳回已經 return 的 do_something 函式中
    longjmp(env, 1);

    return 0;
}
