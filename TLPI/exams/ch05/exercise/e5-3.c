/**
 * 本習題的設計目的在於示範為何以 O_APPEND 旗標開啟檔案來保障操作的原子性是必要的。
 * 請設計程式，可接收多達三個命令列參數：
 * $ e5-3 filename num-bytes [x]
 * 此程式應該開啟指定的檔名（若有需要時建立），並使用 writef）以一次寫入一個位元組
 * （byte）的方式將 num-bytes 資料增加到檔案中。預設時，程式應該以 O_APPEND 旗標
 * 開啟檔案，但是若有提供第三個命令列參數（x），則應該忽略 O_APPEND旗標，並在每次 
 * write（）以前，將程式改為執行lseek（fd, O, SEEK_END） 呼叫。在沒有x參數的情況
 * 下，同時執行此程式的兩個實體（instance），寫入一百萬個位元組到相同的檔案：
 * $ e5-3 f1 1000000 & atomic_append f1 1000000
 * 重複同樣的步驟，寫到不同的檔案，但是這次要設定x參數：
 * $ e5-3 f2 x x& atomic_append f2 1000000 x
 * 使用1-1列出檔案f1及f2的大小，並表達其差異之處。
 * 
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
    
    fd = open(argv[1], O_RDWR | O_APPEND | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1)
        errExit("open");

    off = atoll(argv[2]);
    if (lseek(fd, off, SEEK_SET) == -1)
        errExit("lseek");
    
    if (write(fd, "test1 test1\n", 12) == -1) 
        errExit("write");
    
    exit(EXIT_SUCCESS);
}