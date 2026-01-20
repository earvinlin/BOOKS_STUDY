#if defined(USE_MYLIB_INTEL)
    #include "../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
int
main(int argc, char *argv[])
{
printf("Hello world\n");
write(STDOUT_FILENO, "Ciao\n", 5);
if (fork() == -1)
errExit("fork");
/* Both child and parent continue execution here */
exit(EXIT_SUCCESS);
}