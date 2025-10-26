#define _GNU_SOURCE
#include <sys/utsname.h>
#include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu use
//#include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For mac-mini(m2)'s vmubuntu use

int main(int argc, char *argv[])
{
    struct utsname uts;
    
    if (uname(&uts) == -1)
        errExit("uname");

    printf("Node name: %s\n", uts.nodename);
    printf("System name: %s\n", uts.sysname);
    printf("Release: %s\n", uts.release);
    printf("Version: %s\n", uts.version);
    printf("Machine: %s\n", uts.machine);

#ifdef _GNU_SOURCE
    printf("Domain name: %s\n", uts.domainname);
#endif

    exit(EXIT_SUCCESS);
}