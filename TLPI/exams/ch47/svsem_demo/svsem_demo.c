#include <sys/types.h>
#include <sys/sem.h>
#include <sys/stat.h>
#include "../../tlpi-book/time/curr_time.h" /* Declaration of currTime() */
#include "../../tlpi-book/svsem/semun.h" /* Definition of semun union */
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    int semid;
    if (argc < 2 || argc > 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s init-value\n or: %s semid operation\n", argv[0], argv[0]);

    if (argc == 2) { /* Create and initialize semaphore */
        union semun arg;
        semid = semget(IPC_PRIVATE, 1, S_IRUSR | S_IWUSR);
        if (semid == -1)
            errExit("semid");

        arg.val = getInt(argv[1], 0, "init-value");
        if (semctl(semid, /* semnum= */ 0, SETVAL, arg) == -1)
            errExit("semctl");

            printf("Semaphore ID = %d\n", semid);
    } else { /* Perform an operation on first semaphore */
        struct sembuf sop; /* Structure defining operation */
        semid = getInt(argv[1], 0, "semid");
        sop.sem_num = 0; /* Specifies first semaphore in set */
        sop.sem_op = getInt(argv[2], 0, "operation");

        /* Add, subtract, or wait for 0 */

        sop.sem_flg = 0; /* No special options for operation */
        printf("%ld: about to semop at %s\n", (long) getpid(), currTime("%T"));
        if (semop(semid, &sop, 1) == -1)
            errExit("semop");
        printf("%ld: semop completed at %s\n", (long) getpid(), currTime("%T"));
    }
    exit(EXIT_SUCCESS);
}