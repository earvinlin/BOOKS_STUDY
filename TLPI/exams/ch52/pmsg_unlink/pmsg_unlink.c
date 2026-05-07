#include <mqueue.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
	if (argc != 2 || strcmp(argv[1], "--help") == 0)
		usageErr("%s mq-name\n", argv[0]);

	if (mq_unlink(argv[1]) == -1)
		errExit("mq_unlink");

	exit(EXIT_SUCCESS);
}

