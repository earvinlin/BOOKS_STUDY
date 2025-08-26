#include <stdio.h>
/*
    在 C 語言的巨集（macro）中，#expr 是一種字串化運算子（stringizing operator），
    用來將巨集的參數轉換成字串字面值。這是 C 預處理器的一個強大功能，讓你可以在編譯前把
    程式碼片段變成文字。
*/
#define PRINT_EXPR(expr) printf(#expr " is %d\n", (expr))

int main() 
{
    PRINT_EXPR(3 + 4);     // 輸出：3 + 4 is 7
    PRINT_EXPR(4 * 12);    // 輸出：4 * 12 is 48

    PRINT_EXPR(33 / 3 );    // 輸出：4 * 12 is 48
    PRINT_EXPR(41 - 22);    // 輸出：4 * 12 is 48

    return 0;
}
