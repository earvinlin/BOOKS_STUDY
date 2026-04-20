#include <sys/types.h>
#include <sys/sem.h>
#include "../../tlpi-book/svsem/semun.h" /* Definition of semun union */
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    struct semid_ds ds;
    union semun arg; /* Fourth argument for semctl() */
    int j, semid;

    // tlpi_hdr.h → errExit(), getInt() 等工具
    if (argc < 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s semid val...\n", argv[0]);

    semid = getInt(argv[1], 0, "semid");

    /* Obtain size of semaphore set */

    // 讀取 semaphore set metadata（IPC_STAT）
    arg.buf = &ds;
    if (semctl(semid, 0, IPC_STAT, arg) == -1)
        errExit("semctl");

    if (ds.sem_nsems != argc - 2)
        cmdLineErr("Set contains %ld semaphores, but %d values were supplied\n",
            (long) ds.sem_nsems, argc - 2);

    /* Set up array of values; perform semaphore initialization */

    arg.array = calloc(ds.sem_nsems, sizeof(arg.array[0]));
    if (arg.array == NULL)
        errExit("calloc");

    for (j = 2; j < argc; j++)
        arg.array[j - 2] = getInt(argv[j], 0, "val");

    if (semctl(semid, 0, SETALL, arg) == -1)
        errExit("semctl-SETALL");

    printf("Semaphore values changed (PID=%ld)\n", (long) getpid());
    
    exit(EXIT_SUCCESS);
}