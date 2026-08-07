/**
 * 本習題的設計目的在於示範為何以 O_APPEND 旗標開啟檔案來保障操作的原子性是必要的。
 * 請設計程式，可接收多達三個命令列參數：
 * $ e5-3 filename num-bytes [x]
 * 此程式應該開啟指定的檔名（若有需要時建立），並使用 writef）以一次寫入一個位元組
 * （byte）的方式將 num-bytes 資料增加到檔案中。預設時，程式應該以 O_APPEND 旗標
 * 開啟檔案，但是若有提供第三個命令列參數（x），則應該忽略 O_APPEND旗標，並在每次 
 * write（）以前，將程式改為執行lseek（fd, O, SEEK_END） 呼叫。在沒有x參數的情況
 * 下，同時執行此程式的兩個實體（instance），寫入一百萬個位元組到相同的檔案：
 * $ ./e5-3_arm f1 1000000 & ./e5-3_arm f1 1000000
 * 重複同樣的步驟，寫到不同的檔案，但是這次要設定x參數：
 * $ ./e5-3_arm f2 1000000 x & ./e5-3_arm f2 1000000 x
 * 使用1-1列出檔案f1及f2的大小，並表達其差異之處。
 * 
 * 【編譯指令】
 * -- 以 macos為例，outfile = e5-3_arm --
  gcc e5-3.c \
  -I/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
  -L/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
  -ltlpi -o e5-3_arm
 */
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
    mode_t mode =  S_IRUSR | S_IWUSR;
    int isAppend = 0;

    int flag;
    if (argc == 4 && strcmp(argv[3], "x") == 0)
        flag =  O_RDWR | O_CREAT;
    else {
         flag = O_RDWR | O_APPEND | O_CREAT;
         isAppend = 1;
    }

    fd = open(argv[1], flag, mode);
    if (fd == -1)
        errExit("open");

    int numCnt = 0;
    int writeCnt= atoi(argv[2]);
    while (numCnt < writeCnt) {
        if (isAppend) {
        if (write(fd, "a", 1) == -1) 
            errExit("write");
        } else {
        if (lseek(fd, 0, SEEK_END) == -1)
            errExit("lseek");
        if (write(fd, "a", 1) == -1) 
            errExit("write");            
        }
        numCnt = numCnt + 1;
    }
    
    close(fd);
        
    exit(EXIT_SUCCESS);
}