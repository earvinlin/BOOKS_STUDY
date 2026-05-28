#include "i6d_ucase.h"

int main(int argc, char *argv[])
{
    struct sockaddr_in6 svaddr; // ockaddr_in6 → IPv6 位址結構
    int sfd, j;                 // sfd → socket file descriptor
    size_t msgLen;
    ssize_t numBytes;
    char resp[BUF_SIZE];        // resp[] → 用來接收 server 回傳的資料

    if (argc < 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s host-address msg...\n", argv[0]);

    // 建立 IPv6 UDP socket
    // AF_INET6 → IPv6 ; SOCK_DGRAM → UDP ; 0 → 自動選擇 UDP 協定
    // 若建立失敗 → errExit("socket")
    sfd = socket(AF_INET6, SOCK_DGRAM, 0); /* Create client socket */
    if (sfd == -1)
        errExit("socket");

    // 設定 server 位址
    // sin6_family → IPv6 ; sin6_port → server 的 port（network byte order）;
    memset(&svaddr, 0, sizeof(struct sockaddr_in6));
    svaddr.sin6_family = AF_INET6;
    svaddr.sin6_port = htons(PORT_NUM);

    // 解析使用者輸入的 IPv6 位址
    // inet_pton() 會把字串（例如 2001:db8::1）轉成 binary IPv6 address
    if (inet_pton(AF_INET6, argv[1], &svaddr.sin6_addr) <= 0)
        fatal("inet_pton failed for address '%s'", argv[1]);

    /* Send messages to server; echo responses on stdout */

    for (j = 2; j < argc; j++) {
        msgLen = strlen(argv[j]);

        // sendto()：送出 UDP 封包
        // UDP 是無連線（connectionless），所以每次都要指定 server 位址
        // 若送出的 byte 數不等於 msgLen → fatal("sendto")
        if (sendto(sfd, argv[j], msgLen, 0, (struct sockaddr *) &svaddr,
            sizeof(struct sockaddr_in6)) != msgLen)
            fatal("sendto");

        // recvfrom()：接收 server 回傳的資料
        // 因為 client 不在乎 server 的位址，所以最後兩個參數是 NULL
        // resp 裡會收到 server 回傳的大寫字串
        // 若失敗 → errExit("recvfrom")
        numBytes = recvfrom(sfd, resp, BUF_SIZE, 0, NULL, NULL);
        if (numBytes == -1)
            errExit("recvfrom");

        // 印出結果
        // %.*s 用來印出指定長度的字串（避免未加 \0 的問題）
        printf("Response %d: %.*s\n", j - 1, (int) numBytes, resp);
    }
    exit(EXIT_SUCCESS);
}

/* 整體流程（ASCII 圖）
Client (IPv6 UDP)                         Server (IPv6 UDP)
-----------------                         -------------------
argv[j]  --sendto()--------------------->  recvfrom()
                                          轉大寫
resp   <--sendto()-----------------------  sendto()
印出結果

*/