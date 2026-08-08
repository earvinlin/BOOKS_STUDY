#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/uio.h>

// 假設我們的封包結構
typedef struct {
    int id;
    int data_length;
} Header;

int main() {
    int fd = open("packet.bin", O_RDONLY);
    if (fd < 0) {
        perror("Open failed");
        return 1;
    }

    Header header;
    char payload[100]; // 接收內文的緩衝區

    // 準備 Scatter-Gather I/O 的 iovec 陣列
    struct iovec iov[2];

    // 第一個區塊：讀取 Header
    iov[0].iov_base = &header;
    iov[0].iov_len = sizeof(Header);

    // 第二個區塊：讀取 Payload
    iov[1].iov_base = payload;
    iov[1].iov_len = sizeof(payload) - 1; // 預留 \0 空間

    // 一次呼叫，同時讀取到兩個不同的記憶體區域
    ssize_t bytes_read = readv(fd, iov, 2);

    if (bytes_read < 0) {
        perror("readv failed");
        close(fd);
        return 1;
    }

    payload[bytes_read > sizeof(Header) ? bytes_read - sizeof(Header) : 0] = '\0';

    printf("讀取總 Byte 數: %zd\n", bytes_read);
    printf("Header ID: %d, Data Len: %d\n", header.id, header.data_length);
    printf("Payload 內容: %s\n", payload);

    close(fd);
    return 0;
}