#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int main(int argc, char *argv[])
{
//    char *my_env[] = {"JUICE=peach and apple", NULL};
    char *my_env[] = {"JUICE=peach and apple", "PathA=aaa", NULL};

    execle("diner_info", "diner_info", "5", "11", NULL, my_env);

//    printf("Exec Error: %s.\n", strerror(errno));
    puts(strerror(errno));

    return 0;
}
