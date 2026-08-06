/**
 * 5-2. 寫一個程式，使用 Q_APPEND 旗標開啟一個現有的檔案以供寫入，並接著在寫入一些
 *      資料以前找尋（seek）至檔案的開頭位置。資料會出現在檔案的哪些地方？為什麼？
 * command : ./e5-2_arm t5-2.txt 0
 * 【說明】
 * 要在C語言中使用O_APPEND，需包含<fcntl.h>標頭檔，並將其與O_WRONLY或O_RDWR搭配使用
 * 當呼叫open()加入O_APPEND旗標後：
 *  • 自動跳至末端：作業系統會在每一次呼叫 write()之前，自動將檔案偏移量(File Offset)
 *    移動到當時檔案的最末端。
 *  • 無視 lseek 寫入：即使你在寫入前呼叫 lseek()將指標移到檔案開頭或中間，接下來呼叫
 *     write()時，系統依然會忽略該位置，強制在檔案末端寫入。
 *  • 原子性(Atomic Operation)：在多行程(Multi-process)或多執行緒(Multi-thread)
 *    同時寫入同一個檔案(如 Log 檔)時，核心會確保「移動指標 + 寫入資料」這兩個動作是一
 *    氣呵成的，不會發生資料相互覆蓋(Race Condition)的情況。
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

