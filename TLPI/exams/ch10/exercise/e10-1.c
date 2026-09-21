/**
    假設系統呼叫 sysconf(_SC_CLK_TCK) 的傳回值是 100。假設 times() 傳回的 
    clock_t 值是一個有號誌的 32 位元整數，需要多久這個值才會進入下一個從 0 
    開始的週期呢？請對 clock() 傳回的 CLOCKS_PER_SEC 值進行同樣的運算。
    這其實是在問 計時器計數值（tick counter）溢位（wrap around）週期。
 */

 #include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <grp.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{

/*
    1. times() 的情況
        已知 sysconf(_SC_CLK_TCK) = 100
        表示：1 秒 = 100 ticks
        即每個 tick：1/100 秒 = 10 ms
        ．clock_t 為有號 32 位元
            有號 32 位元整數範圍：−2^31 ~ 2^31−1
            若考慮從 0 開始增加到下一次回到 0 的完整週期：
            2^32 = 4294967296 ticks
            換算秒數 : 42949672.96 ÷ 86400 ≈ 497.10 天
            換算天數 : 42949672.96 ÷ 86400 ≈ 497.10 天
            結果 : 約 42,949,673 秒 ≈ 497.1 天 ≈ 1.36 年
            所以 : times() 的 clock_t 若以 100 ticks/sec 計算，大約 497 天會繞回一圈。

    2. clock() 的情況
        ．CLOCKS_PER_SEC  1000000 : 1 秒 = 1,000,000 clock ticks
            完整週期仍然是：2^32 = 4294967296 ticks
            換算秒數 : 4294967296 ÷ 1000000 = 4294.967296 秒
            換算分鐘 : 4294.967296 ÷ 60 ≈ 71.58 分鐘
            換算小時 : 71.58 ÷ 60 ≈ 1.19 小時
            所以：clock() 若使用 32 位元有號 clock_t 且 CLOCKS_PER_SEC = 1000000，
                 大約 71.6 分鐘就會溢位一圈。
        ．一般公式 : 
            若：clock_t 為 N 位元，頻率為 F ticks/sec
            則完整循環時間：T = 2^N / F
        ．times()
            T = 2^32 / 100 = 42949672.96 秒 ≈ 497 天
        ．clock()
            T = 2^32 / 1000000 = 4294.967296 秒 ≈ 71.6 分鐘
            因此答案為：
            ===========================================
            函式        Tick Rate           32-bit 週期
            ===========================================
            times()     100 ticks/s         約 497 天
            clock()     1,000,000 ticks/s   約 71.6 分鐘
            ※ 這也是為什麼早期使用 32 位元 clock_t 時，clock() 
              很容易在長時間執行程式中發生 wrap-around，而 
              times() 則通常要一年多才會遇到一次。
*/

    return 0;
}