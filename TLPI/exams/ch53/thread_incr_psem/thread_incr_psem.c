#include <semaphore.h>
#include <pthread.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

static int glob = 0;
static sem_t sem;   // sem 是 POSIX semaphore，用來保護 glob
static void * /* Loop 'arg' times incrementing 'glob' */

// 每個 thread 要執行 loops 次 glob++。
threadFunc(void *arg) {
    int loops = *((int *) arg);
    int loc, j;

    for (j = 0; j < loops; j++) {
        // 進入迴圈，每次都要進入臨界區 ; 避免競速條件
        // 如果 semaphore 值 > 0 → 減 1 → 進入臨界區
        // 如果 semaphore 值 == 0 → 阻塞等待
        // 因為初始值是 1，所以一次只有一個 thread 能進入
        if (sem_wait(&sem) == -1)
            errExit("sem_wait");
        // 臨界區：讀取、修改、寫回 glob
        // 這三行是 非原子操作，如果沒有 semaphore 保護，兩個 thread 會互相覆蓋，造成 race condition。
        loc = glob;
        loc++;
        glob = loc;

        // 離開臨界區
        // 把 semaphore 值 +1
        // 如果有 thread 在 sem_wait() 阻塞 → 喚醒其中一個
        // 這樣下一個 thread 才能進入臨界區。
        if (sem_post(&sem) == -1)
            errExit("sem_post");
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t t1, t2;
    int loops, s;

    loops = (argc > 1) ? getInt(argv[1], GN_GT_0, "num-loops") : 10000000;

    /* Initialize a thread-shared mutex with the value 1 */

    // 初始化 semaphore
    // 第二個參數 0 → 表示「thread-shared」，不是 process-shared
    // 初始值 1 → 表示一次允許一個 thread 進入臨界區（mutex 行為）
    if (sem_init(&sem, 0, 1) == -1)
        errExit("sem_init");

    /* Create two threads that increment 'glob' */

    s = pthread_create(&t1, NULL, threadFunc, &loops);
    if (s != 0)
        errExitEN(s, "pthread_create");

    s = pthread_create(&t2, NULL, threadFunc, &loops);
    if (s != 0)
        errExitEN(s, "pthread_create");

    /* Wait for threads to terminate */

    s = pthread_join(t1, NULL);
    if (s != 0)
        errExitEN(s, "pthread_join");

    s = pthread_join(t2, NULL);
    if (s != 0)
        errExitEN(s, "pthread_join");

    printf("glob = %d\n", glob);

    exit(EXIT_SUCCESS);
}
