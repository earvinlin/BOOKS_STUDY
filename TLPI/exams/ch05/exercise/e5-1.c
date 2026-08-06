/**
 * command : ./e5-1_arm test_large.dat 5368709120
 * 【說明】
 * 如果沒有「#define _LARGEFILE64_SOURCE」或於編譯時加上「-D_LARGEFILE64_SOURCE」
 * 在32位元的環境下應該會有問題
 */
#define _LARGEFILE64_SOURCE
#include <sys/stat.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    int fd;
    off_t off;

    if (argc != 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s pathname offset\n", argv[0]);
    
    fd = open(argv[1], O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1)
        errExit("open");

    off = atoll(argv[2]);
    if (lseek(fd, off, SEEK_SET) == -1)
        errExit("lseek");
    
    if (write(fd, "test", 4) == -1) 
        errExit("write");
    
    exit(EXIT_SUCCESS);
}
