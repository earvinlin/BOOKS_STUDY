/*
    編譯指令：gcc e10.c -o e10 -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("請輸入一個數值作為參數。\n");
        return 1;
    }

    double value = atof(argv[1]);  // 將字串轉為浮點數
    printf("The square root of %f is %f\n", value, sqrt(value));
    return 0;
}
