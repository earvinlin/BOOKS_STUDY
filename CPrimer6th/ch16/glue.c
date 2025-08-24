// glue.c -- use the ## operator
#include <stdio.h>
#include <string.h>
#define XNAME(n) x ## n
#define PRINT_XN(n) printf("x" #n " = %d\n", x ## n);
# define X_4 x4

int main(void)
{
	int XNAME(1) = 14; // becomes int x1 = 14;
	int XNAME(2) = 20; // becomes int x2 = 20;
	int x3 = 30;
	PRINT_XN(1); // becomes printf("x1 = %d\n", x1);
	PRINT_XN(2); // becomes printf("x2 = %d\n", x2);
	PRINT_XN(3); // becomes printf("x3 = %d\n", x3);

	int xx = 10;
	printf("test " ", here is %d.\n", xx);

/*	C++ 才支援
	string name = "earvin";
	string ss = "Hello, " name " myfriend^^";
	printf(ss);
*/

	return 0;
}
