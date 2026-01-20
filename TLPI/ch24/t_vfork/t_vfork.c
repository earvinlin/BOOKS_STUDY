#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    int istack = 222;

    switch (vfork()) {
        case -1:
            errExit("vfork");
        case 0: /* Child executes first, in parent's memory space */
            sleep(3); /* Even if we sleep for a while,
                            parent still is not scheduled */
            write(STDOUT_FILENO, "Child executing\n", 16);
            istack *= 3; /* This change will be seen by parent */
            _exit(EXIT_SUCCESS);
        default: /* Parent is blocked until child exits */
            write(STDOUT_FILENO, "Parent executing\n", 17);
            printf("istack=%d\n", istack);
            exit(EXIT_SUCCESS);
    }
}