#include "inet_sockets.h" /* Declares our socket functions */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#define SERVICE "echo" /* Name of UDP service */
#define BUF_SIZE 500 /* Maximum size of datagrams that can be read by client and server */