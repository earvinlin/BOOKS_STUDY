#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>   // open(), O_CREAT, O_WRONLY
#include <unistd.h>  // close()
#include <sys/stat.h> // S_IRUSR, S_IWUSR 權限巨集
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#include <libgen.h> // 使用 basename() 必須引入此標頭檔
#include <string.h> // 使用 strrchr()

#define BUF_SIZE 100  // 每次最多讀取 128 位元組


char * getProgName(char *s) {
    // 尋找字串中最後一個 '/'
    char *progName = strrchr(s, '/');

    if (progName != NULL) {
        progName++; // 跳過 '/' 本身，指向檔名開頭
    } else {
        progName = s; // 如果路徑中沒有 '/'，直接使用 argv[0]
    }
//    printf("僅取程式名稱 : %s\n", progName);
    return progName;
}

int main(int argc, char *argv[]) {
    int infd, outfd, ap;
    char buffer[BUF_SIZE];
    char outflname[10]; // 用來存放生成的字串空間 (記得要留空間給 '\0')
    char prefix = 'e';
    ssize_t bytes_read;
    char * op = argv[2];
    long sparseFileSize = 0;
    char * writeStr;
//    size_t len;
    off_t offset;

    /* 
        argv的資料結構本質上是指向陣列的指標
        ex: ./splitFileTest spfile.txt 100
        argv[0] ""./splitFileTest"  程式本身的名稱/路徑（永遠是第一個參數）
        argv[1] "spfile.txt"        使用者傳入的第 1 個參數
        argv[2] "100"               使用者傳入的第 2 個參數
    */
    printf("The argc value is %d\n", argc);
    if (argc < 3) {
        char *prog_name = basename(argv[0]);    // 使用 basename 取得不帶路徑的純檔名
        printf("parameter error, program is %s\n", prog_name);
        exit(1);
    }

    // 讀取要分割的檔案
    infd = open(argv[1], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (infd == -1) {
        perror("open fail! (for read)");
        return 1;
    }
    printf("成功開啟檔案，檔案描述符(FD)為：%d\n", infd);
    
    uint8_t i = 0;
    while ((bytes_read = read(infd, buffer, BUF_SIZE)) > 0) {
        snprintf(outflname, sizeof(outflname), "%c%02d", prefix, i++);
        outfd = open(outflname, O_WRONLY | O_CREAT | O_TRUNC, 
                S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if (outfd == -1) {
            perror("open fail!");
            return 1;
        }
        printf("成功建立檔案，檔案描述符(FD)為：%d\n", outfd);     

        // 寫檔
        if (write(outfd, buffer, bytes_read) != bytes_read) {
            perror("write 寫入異常");
            close(outfd);
            return 1;
        }
        if (bytes_read == -1) {
            perror("read 讀取錯誤");
        }
        close(outfd);
        printf("\n資料已成功寫入 %s！\n", outflname);
    }
    if (bytes_read == -1) {
        perror("read 發生錯誤");
    } else {
        printf("\n=== 檔案讀取完畢 (EOF) ===\n");
    }
    close(infd);  // 關閉檔案

    return 0;
}