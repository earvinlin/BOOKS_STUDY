#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    int flags;

    if (argc > 1) {
        flags = fcntl(STDOUT_FILENO, F_GETFD); /* Fetch flags */

        if (flags == -1)
            errExit("fcntl - F_GETFD");

        flags |= FD_CLOEXEC; /* Turn on FD_CLOEXEC */
        if (fcntl(STDOUT_FILENO, F_SETFD, flags) == -1) /* Update flags */
            errExit("fcntl - F_SETFD");
    }
    execlp("ls", "ls", "-l", argv[0], (char *) NULL);

    errExit("execlp");
}