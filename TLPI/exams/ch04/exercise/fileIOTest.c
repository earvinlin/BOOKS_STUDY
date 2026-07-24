#include <stdio.h>
#include <fcntl.h>   // open(), O_CREAT, O_WRONLY
#include <unistd.h>  // close()
#include <sys/stat.h> // S_IRUSR, S_IWUSR 權限巨集
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

#define MAX_READ 20

int main(int argc, char *argv[]) {
    char buffer[MAX_READ];

    // 建立新檔案，並設定權限為 0644 (所有者可讀寫，其他人唯讀)
    int fd = open("fileIOTest.txt", O_WRONLY | O_CREAT | O_TRUNC, 
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

    if (fd == -1) {
        perror("open fail!");
        return 1;
    }
    printf("成功建立檔案，檔案描述符(FD)為：%d\n", fd);

    // 呼叫 write 寫入檔案
    char text[10] ="work work";
//    const char *text = "test test\n";
    size_t len = 10;
    ssize_t bytes_written = write(fd, text, len);

    // 防呆：檢查是否寫入成功且長度相符
    if (bytes_written == -1) {
        perror("write 到檔案失敗");
        close(fd);
        return 1;
    } else if ((size_t)bytes_written != len) {
        fprintf(stderr, "警告：未完整寫入資料！預期 %zu，實際寫入 %zd\n", len, bytes_written);
    } else {
        printf("成功寫入 %zd 位元組到 test_write.txt\n", bytes_written);
    }

    close(fd);  // 關閉檔案

    return 0;
}