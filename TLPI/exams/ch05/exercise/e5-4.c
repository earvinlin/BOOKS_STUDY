/**
 * 使用fcntl() 實作dup() 及 dup2()，並在有需要之處執行close()(你可以忽略dup2()
 * 及 fcntl() 實際上對於某些錯誤的例子會傳回不一樣的errno 值)。對於dup2()，記得要
 * 處理特殊的例子(在oldfd 等於newfd 之處)。在此例中，你您該檢查oldfd 是否為有效值，
 * 例如：檢查 fcntlfoldfd, F_GETFL） 是否成功。若 oldfd 不是有效值，那麼函式應該
 * 傳回 -1 並將 errno 設定為 EBADF。
 */
#include <sys/stat.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

/**
    【說明】
    dup() 的核心功能就是「取得一個新的檔案描述符，指向與 oldfd 相同的 open file description」。
    它不會重新開檔、不會改變檔案偏移量、不會改變 flags，只是讓同一個檔案多一個 fd 入口。
    
    🧱 實作重點（你在題目中提到的要求逐項對應）
    ．檢查 oldfd 是否有效  
        → fcntl(oldfd, F_GETFL) 成功代表 oldfd 是有效的開啟檔案描述符。
    ．dup() 使用 fcntl(F_DUPFD)  
        → fcntl(oldfd, F_DUPFD, 0) 會回傳最小可用的 fd。
    ．dup2() 處理 oldfd == newfd  
        → 直接回傳 newfd，不做 close、不做 fcntl。
    ．dup2() 需要 close(newfd)  
        → 若 newfd 已存在，必須先關閉，否則 fcntl(F_DUPFD, newfd) 不會覆蓋它。
    ．忽略某些錯誤時 errno 不同的情況  
        → 題目允許你忽略 fcntl 與 dup2 在某些錯誤時 errno 不一致的細節。
 */
int my_dup(int oldfd) {
    /* 檢查 oldfd 是否有效 */
    if (fcntl(oldfd, F_GETFL) == -1) {
        errno = EBADF;
        return -1;
    }
    /* 使用 fcntl(F_DUPFD) 產生新的 fd */
    return fcntl(oldfd, F_DUPFD, 0);
}

int my_dup2(int oldfd, int newfd) {
    /* 檢查 oldfd 是否有效 */
    if (fcntl(oldfd, F_GETFL) == -1) {
        errno = EBADF;
        return -1;
    }
    /* 特殊案例：oldfd == newfd */
    if (oldfd == newfd) {
        /* oldfd 已經是有效 fd，因此直接回傳即可 */
        return newfd;
    }
    /* 若 newfd 已開啟，先關閉它 */
    close(newfd);
    /* 使用 fcntl(F_DUPFD) 指定 newfd 作為最低可用 fd */
    return fcntl(oldfd, F_DUPFD, newfd);
}


int main(int argc, char *argv[])
{
    int old_fd,new_fd;
    int flag =  O_RDWR | O_CREAT;
    mode_t mode =  S_IRUSR | S_IWUSR;

    old_fd = open(argv[1], flag, mode);
    if (old_fd == -1)
        errExit("open");
    printf("fd open %d success.\n", old_fd);        

    if (write(old_fd, "a", 1) == -1) 
        errExit("write");
    if (write(old_fd, "\n", 1) == -1) 
        errExit("write");

    exit(EXIT_SUCCESS);
}


/**
    -- 之前寫的版本…亂~~ (20260808) --
    int fd, dup_fd;
    int fd2, dup2_fd;
    int fcntl_fd;
    int flag =  O_RDWR | O_CREAT;
    mode_t mode =  S_IRUSR | S_IWUSR;

    // 以fcnt() 實作 dup()
   fd = open(argv[1], flag, mode);
    if (fd == -1)
        errExit("open");
    else {
        if (write(fd, "a", 1) == -1) 
            errExit("write");
        printf("fd open %d success.\n", fd);

        dup_fd = dup(fd);
        if (dup_fd == -1)
            errExit("error dup_fd");
        printf("dup_fd open %d success.\n", dup_fd);

        if (write(dup_fd, "b", 1) == -1) 
            errExit("write");

        // use fcntl()
        fcntl_fd = fcntl(fd, F_DUPFD, 100);
        if (fcntl_fd == -1)
            errExit("error fcntl");

        printf("fcntl_fd open %d success.\n", fcntl_fd);
        if (write(fcntl_fd, "c", 1) == -1) 
            errExit("write");
    }

    // 以fcnt() 實作 dup2()
    fd2 = open(argv[2], flag, mode);
    if (fd2 == -1)
        errExit("open fd2");
    printf("fd2 open %d success.\n", fd2);

    if (write(fd2, "A", 1) == -1) 
        errExit("write");
       
        
    dup2_fd = dup2(fd2, dup_fd);
    if (dup2_fd == -1)
        errExit("error dup2_fd");
    
    printf("dup2_fd open %d success.\n", dup2_fd);

    // 檢查 dup_fd是否合法且存在
    if (fcntl(dup_fd, F_GETFD) != -1) {
        printf("dup_fd 存在");
    } else {
        // 如果回傳 -1 且 errno 為 EBADF，代表 FD 不存在或已關閉
        if (errno == EBADF) {
            printf("dup_fd 不存在");
        }
    }

    if (write(dup_fd, "A", 1) == -1) 
        errExit("write");
   
    // use fcntl()
//        fcntl_fd = fcntl(fd, F_DUPFD, 100);
//        if (fcntl_fd == -1)
//            errExit("error fcntl");    


    close(fd);
    close(dup_fd);
    close(dup2_fd);
    close(fcntl_fd);
 */