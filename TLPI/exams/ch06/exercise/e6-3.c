/**
 * 使用getenv()函式、putenv()函式，必要時可直接修改 nviron，來實作setenv()函式和
 * unsetenv(函式。此處的unsetenv()函式應檢查是否對環境變數進行了多次定義，如果是多
 * 次定義則將移除對該變數的全部定義(glibc版本的unsetenv()函式實作了這一功能)。
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif


int main(int argc, char *argv[])
{


    return 0;
}
