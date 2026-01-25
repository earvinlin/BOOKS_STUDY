#include <stdlib.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    printf("Initial value of USER: %s\n", getenv("USER"));

    if (putenv("USER=britta") != 0)
        errExit("putenv");

    execl("/usr/bin/printenv", "printenv", "USER", "SHELL", (char *) NULL);
    
    errExit("execl"); /* If we get here, something went wrong */
}