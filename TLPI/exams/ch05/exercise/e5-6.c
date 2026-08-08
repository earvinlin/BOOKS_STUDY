/**
 * 在下列的程式碼中，每次呼叫 write() 之後，表達輸出檔案的內容會是什麼，以及為什麼：
 * fd1 = open(file, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
 * fd2 = dup(fd1);
 * fd3 = open(file, O_RDWR);
 * write(fd1, "Hello,", 6);
 * write(fd2, " world", 6);
 * lseek (fd2, 0, SEEK_SET);
 * write(fd1, "HELLO,", 6);
 * write(fd3, "Gidday", 6);
 */
#include <sys/stat.h>
#include <fcntl.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

void print_file(char *read_file) {
    int fd;
    char buffer[256];
    ssize_t bytesRead;
    
    fd = open(read_file, O_RDONLY);
    if (fd == -1) {
        perror("檔案開啟失敗");
        return;
    }

    // 從檔案讀取資料
    while ((bytesRead = read(fd, buffer, sizeof(buffer)-1)) > 0) {
        buffer[bytesRead] = '\0';  // 加上字串結尾
        printf("%s", buffer);
    }

    if (bytesRead == -1) {
        perror("讀取失敗");
    }

    close(fd);
    return;
}

// 查詢開啟檔案目前的檔案偏移值
off_t get_file_offset(int fd) {
    // 從「目前位置 (SEEK_CUR)」移動 0 位元組
    // lseek 會傳回當前的絕對偏移位置
    off_t current_offset = lseek(fd, 0, SEEK_CUR);

    if (current_offset == (off_t) -1) {
        perror("查詢檔案偏移值失敗 (lseek error)");
    }

    return current_offset;
}

int main(int argc, char *argv[])
{
    int fd1, fd2, fd3;

    fd1 = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd1 == -1)
        errExit("open");
    printf("fd1 open %d success.\n", fd1);        

    fd2 = dup(fd1);
    fd3 = open(argv[1], O_RDWR);

    if (write(fd1, "Hello,", 6) == -1) 
        errExit("write");
    printf("fd1 current cursor positon : %lld\n", (long long) get_file_offset(fd1));

    if (write(fd2, " world", 6) == -1) 
        errExit("write");
    printf("fd2 current cursor positon : %lld\n", (long long) get_file_offset(fd2));

    print_file(argv[1]);
    printf("\n");

    lseek (fd2, 0, SEEK_SET);
    
    if (write(fd2, "HELLO,", 6) == -1) 
        errExit("write");
    printf("fd2 current cursor positon : %lld\n", (long long) get_file_offset(fd2));

    print_file(argv[1]);
    printf("\n");

    if (write(fd3, "Gidday", 6) == -1) 
    errExit("write");
    printf("fd3 current cursor positon : %lld\n", (long long) get_file_offset(fd3));

    print_file(argv[1]);
    printf("\n");
    
    exit(EXIT_SUCCESS);
}
