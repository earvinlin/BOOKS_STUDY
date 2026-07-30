/*
  撰寫產生器程式 (readv_test_data.c)
  
  編譯與執行
  gcc readv_test_data.c -o generate_packet
  ./generate_packet
  --> 執行後，當前目錄就會產生一個 packet.bin 測試檔！
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// 必須與讀取端的 Header 結構一模一樣
typedef struct {
    int id;
    int data_length;
} Header;

int main() {
    // 1. 準備 Header 資料
    Header header;
    header.id = 1001;                     // 假設封包 ID 是 1001
    
    // 2. 準備 Payload 資料
    char payload[] = "Hello, Scatter-Gather I/O!";
    header.data_length = strlen(payload); // 記錄 Payload 的長度

    // 3. 以「寫入 + 建立」模式開啟 packet.bin 檔案
    // O_WRONLY: 唯讀寫 / O_CREAT: 檔案不存在則建立 / O_TRUNC: 若存在則清空
    int fd = open("packet.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("開啟/建立 packet.bin 失敗");
        return 1;
    }

    // 4. 先寫入 Header（二進位）
    write(fd, &header, sizeof(Header));

    // 5. 再寫入 Payload（資料內容）
    write(fd, payload, strlen(payload));

    close(fd);
    printf("成功產生 packet.bin 檔案！\n");
    return 0;
}