/**
 * 顯示行程環境
 */
#define _GNU_SOURCE /* To get various declarations from <stdlib.h> */
#include <stdlib.h>
#include <unistd.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

extern char **environ;

int main(int argc, char *argv[])
{
    int j;
    char **ep;

    // 擦除目前行程（Process）中的所有環境變數，將全域變數 environ 清空（指向 NULL）
    clearenv(); /* Erase entire environment */
    
    for (j = 1; j < argc; j++)
        /**
            將使用者輸入的參數（格式需為 NAME=VALUE）直接放進環境變數清單中
            NOTE :
            putenv() 會直接將傳入字串的指標存入 environ，而不會複製一份字串。
            因此傳入的字串不能是區域變數（Stack），否則會導致 Undefined Behavior
            （這裡傳入的是 argv，在整支程式執行期間均有效，所以安全）       
         */
        if (putenv(argv[j]) != 0)
            errExit("putenv: %s", argv[j]);
    
    /**
        設定環境變數 GREET 的值為 "Hello world"
        第 3 個參數 0 (overwrite 旗標) :
        - 若設為 0：如果 GREET 已經存在，則不更新其內容；若不存在則新增
        - 若設為 1：無論是否存在，都強制覆蓋。
     */
    if (setenv("GREET", "Hello world", 0) == -1)
        errExit("setenv");
    
    /**
        刪除環境變數
        從環境變數清單中移除名為 BYE 的變數。若該變數原本就不存在，函式也會順利返回 0
    */
    unsetenv("BYE");
    
    /**
        走訪並印出
        利用 C Standard Library 提供的全域指標陣列 extern char **environ，逐一走
        訪並印出目前所有的環境變數
     */
    for (ep = environ; *ep != NULL; ep++)
        puts(*ep);

    exit(EXIT_SUCCESS);
}
