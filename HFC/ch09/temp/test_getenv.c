#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
    char *env[] = {"JUICE=peach and apple","AGE=14",  NULL};

//    execle("diner_info", "diner_info", "4", NULL, my_env);
    execle("/usr/bin/env", "env", NULL, env);

//	printf("Diners: %s\n", argv[1]);
//	printf("Juice: %s\n", getenv("JUICE"));
    perror("execle failed");
    
	return 0;
}

