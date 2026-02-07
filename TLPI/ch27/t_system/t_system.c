#include <sys/wait.h>
#include "print_wait_status.h"
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define MAX_CMD_LEN 200
int main(int argc, char *argv[])
{
    char str[MAX_CMD_LEN]; /* Command to be executed by system() */
    int status; /* Status return from system() */

    for (;;) { /* Read and execute a shell command */
        printf("Command: ");

        fflush(stdout);
        if (fgets(str, MAX_CMD_LEN, stdin) == NULL)
            break; /* end-of-file */

        status = system(str);
        printf("system() returned: status=0x%04x (%d,%d)\n",
            (unsigned int) status, status >> 8, status & 0xff);

        if (status == -1) {
            errExit("system");
        } else {
            if (WIFEXITED(status) && WEXITSTATUS(status) == 127)
                printf("(Probably) could not invoke shell\n");
            else /* Shell successfully executed command */
                printWaitStatus(NULL, status);
        }
    }
    
    exit(EXIT_SUCCESS);
}