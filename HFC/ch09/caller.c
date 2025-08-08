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
    if (execle("diner_info", "diner_info", "4", NULL, my_env) == -1) {
        fprintf(stderr, "Cannot run ipconfig: %s\n", strerror(errno));
        return 1;
    }

    return 0;
}