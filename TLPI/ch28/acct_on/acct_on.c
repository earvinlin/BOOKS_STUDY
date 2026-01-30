//#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#include <unistd.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    if (argc > 2 || (argc > 1 && strcmp(argv[1], "--help") == 0))
        usageErr("%s [file]\n");

    if (acct(argv[1]) == -1)
        errExit("acct");

    printf("Process accounting %s\n", (argv[1] == NULL) ? "disabled" : "enabled");

    exit(EXIT_SUCCESS);
}