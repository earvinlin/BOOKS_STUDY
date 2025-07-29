#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
<<<<<<< HEAD:HFC/ch09/test_diner_info.c
    char *my_env[] = {"JUICE=peach and apple", NULL};

    execle("diner_info", "diner_info", "4", NULL, my_env);
=======
//    char *my_env[] = {"JUICE=peach and apple", NULL};
//    execle("diner_info", "diner_info", "4", NULL, my_env);
>>>>>>> 98bc58c6248c069412eb2f6961ddffe87cfe0e8c:HFC/ch09/test_getenv.c

    return 0;
}
