/**
 * gcc -c encrypt.c -o encrypt.o
 * gcc -c checksum.c -o checksum.o
 * gcc test_code.c encrypt.o checksum.o -o test_code
 */

#include <stdio.h>
//#include "encrypt.h"
//#include "checksum.h"
#include <encrypt.h>
#include <checksum.h>

int main()
{
	char s[] = "Speak friend and enter";
	encrypt(s);
	printf("Encrypted to '%s'\n", s);
	printf("Checksum is %i\n", checksum(s));
	encrypt(s);
	printf("Decrypted back to '%s'\n", s);
	printf("Checksum is %i\n", checksum(s));

	return 0;
}

