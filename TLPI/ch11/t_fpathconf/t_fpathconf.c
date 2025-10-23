#include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu use
//#include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For mac-mini(m2)'s vmubuntu use

/* Print 'msg' plus value of fpathconf(fd, name) */
static void fpathconfPrint(const char *msg, int fd, int name) {
    long lim;
    errno = 0;
    lim = fpathconf(fd, name);

    if (lim != -1) { /* Call succeeded, limit determinate */
        printf("%s %ld\n", msg, lim);
    } else {
        if (errno == 0) /* Call succeeded, limit indeterminate */
            printf("%s (indeterminate)\n", msg);
        else /* Call failed */
            errExit("fpathconf %s", msg);
    }
}

int main(int argc, char *argv[])
{
    fpathconfPrint("_PC_NAME_MAX: ", STDIN_FILENO, _PC_NAME_MAX);
    fpathconfPrint("_PC_PATH_MAX: ", STDIN_FILENO, _PC_PATH_MAX);
    fpathconfPrint("_PC_PIPE_BUF: ", STDIN_FILENO, _PC_PIPE_BUF);

    exit(EXIT_SUCCESS);
}