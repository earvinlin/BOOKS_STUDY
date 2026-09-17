#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

/*
    struct timeval {
        time_t      tv_sec;     // 自 Epoch (1970/1/1) 起算經過的秒數 (Seconds)
        suseconds_t tv_usec;    // 微秒數 (Microseconds, 範圍 0 ~ 999,999)
    };
*/
int main(int argc, char *argv[]) 
{
    struct timeval start, end;
    // 將目前執行緒（Thread）或行程（Process）暫停執行指定微秒（Microseconds，即10^-6秒）
    // 的系統函式 (usec : 要暫停的微秒數 (1 秒 = 1,000,000 微秒))
    // 成功回傳 0；失敗回傳 -1 並設定 errno。
    // 已被 POSIX 廢棄：POSIX.1-2001 已將 usleep() 標記為廢棄(POSIX.1-2008 更是將其完全移除)
    usleep(5000000);
    
    // 是 POSIX / Linux 系統程式設計中的系統呼叫，主要用於取得微秒（Microsecond，即10^-6秒）
    // 層級的高精度當前時間。
    // 與僅精確到「秒」的 time() 相比，gettimeofday() 常用於測量效能、計算程式執行時間或紀錄
    // 高精度日誌。
    // 成功回傳 0；失敗回傳 -1 並設定 errno。
    gettimeofday(&end, NULL);

    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    printf("開始秒數：%ld, 微秒：%ld\n", start.tv_sec, start.tv_usec);
    printf("結束秒數：%ld, 微秒：%ld\n", end.tv_sec, end.tv_usec);
    printf("總共耗時：%.6f 秒（或%.2f毫秒）\n", elapsed, elapsed * 1000.0);

    //-- time() usage --//
    /**
        time_t time(time_t *tloc);
        • tloc：指向 time_t 型態變數的指標。
            • 若傳入非 NULL 指標：函式會將結果寫入該指標指向的記憶體，並同時傳回該值。
            • 若傳入 NULL：函式僅透過傳回值回傳時間戳記（最常見用法）。
        • 傳回值：
            • 成功：傳回自 1970 年 1 月 1 日 00:00:00 UTC（Unix Epoch） 起算至目前的總秒數。
            • 失敗：傳回 (time_t)-1，並設定 errno。
        • 注意：
            • 精度限制：最低精度僅到 1 秒（若需要微秒或奈秒等級，請改用 gettimeofday() 或 
                      clock_gettime()）。
            • 系統時間依賴：取得的是牆上時間（Wall-clock time），會受到系統時間校正（如 NTP）
                          或手動修改時間影響。
        • 應用場景：time() 最常用於取得當前秒數，或做為隨機數種子（srand()）
    */
    time_t now = time(NULL);    // 直接接受傳回值
    // time(&now);              // 另一種寫法，將結果寫入 now 變數
    if (now == (time_t)-1) {
        perror("time() failed");
        return EXIT_FAILURE;
    }
    printf("當前Unix時間戳記：%ld 秒\n", (long)now);
    printf("當前本地時間：%s", ctime(&now));

    srand((unsigned int)time(NULL)); // 使用當前時間作為隨機數種子
    printf("隨機數範例：%d\n", rand() % 100); // 產生 0~99 的隨機數

    return 0;
}