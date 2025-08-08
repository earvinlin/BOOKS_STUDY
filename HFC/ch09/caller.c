#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int main()
{
    char *my_env[] = {"JUICE=nothing", NULL};
/*
    execle("diner_info", "diner_info", 4, NULL, my_env);
//    puts("Dude - the diner_info code must be busted");
    puts(strerror(errno));
*/

/*
    if (execle("diner_info", "diner_info", "4", NULL, my_env) == -1) {
        fprintf(stderr, "Cannot run ipconfig: %s\n", strerror(errno));
        return 1;
    }
*/

#if defined(_WIN32)
    printf("Windows (32-bit or 64-bit)\n");

#elif defined(_WIN64)
    printf("Windows 64-bit\n");

#elif defined(__linux__)
    printf("Linux\n");
    if (execle("diner_info_linux", "diner_info_linux", "4", NULL, my_env) == -1) {
        fprintf(stderr, "Cannot run ipconfig: %s\n", strerror(errno));
        return 1;
    }

#elif defined(__APPLE__) && defined(__MACH__)
    printf("macOS\n");
    printf("Linux\n");
    if (execle("diner_info_mac", "diner_info_mac", "4", NULL, my_env) == -1) {
        fprintf(stderr, "Cannot run ipconfig: %s\n", strerror(errno));
        return 1;
    }

#elif defined(__unix__)
    printf("Unix (generic)\n");

#else
    printf("Unknown OS\n");

#endif

    return 0;
}