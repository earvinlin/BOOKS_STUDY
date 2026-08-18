#include <setjmp.h>
//#include <unistd.h>
#include "../../tlpi-book/mylib/tlpi_hdr.h"

static jmp_buf env;

static void f2(void) {
    longjmp(env, 2);
}

static void f1(int argc) {
    if (argc == 1)
        longjmp(env, 1);
    
    printf("Will exec f2()!\n");
    f2();
}

int main(int argc, char *argv[]) {
    switch (setjmp(env)) {
        case 0: /* This is the return after the initial setjmp() */
            printf("Calling f1() after initial setjmp()\n");
            f1(argc); /* Never returns... */
            printf("case 0, will break!\n"); // 永遠執行不到！！！
            break; /* ... but this is good form */

        case 1:
        /**
            2.	情況 A：不帶任何參數執行（argc == 1，只有程式名稱本身）
            •	進入 f1() 後，條件 argc == 1 成立，執行 longjmp(env, 1)。
            •	CPU 狀態直接還原並飛回 main() 中的 setjmp() 位置，此時 setjmp() 的「偽傳回值」為 1。
            •	進入 case 1:，印出 "We jumped back from f1()\n"，最後順利結束程式。
        */
            printf("We jumped back from f1()\n");
            break;

        case 2:
        /**
            3.	情況 B：帶有參數執行（argc > 1，例如 ./app arg1）
            •	進入 f1() 後，跳過 if 判斷，進入並呼叫 f2()。
            •	f2() 內部執行 longjmp(env, 2)。
            •	CPU 跨越兩層函式呼叫（直接剝離 f2() 與 f1() 的 Stack Frames），飛回 main() 中的 setjmp()，此時「偽傳回值」為 2。
            •	進入 case 2:，印出 "We jumped back from f2()\n"，最後結束程式。
        */
            printf("We jumped back from f2()\n");
            break;
    }

    exit(EXIT_SUCCESS);
}