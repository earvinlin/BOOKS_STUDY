#include "inet_sockets.h" /* Declares our socket functions */
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    // 1. 角色配置：三個關鍵的檔案描述符（File Descriptor）
    // listenFd（監聽總機）：代表伺服器用來「等待別人連線」的監聽路徑。
    // connFd（客戶端電話機）：代表「主動撥號出去」的客戶端。
    // acceptFd（伺服器專線）：當總機 listenFd 接到客戶端的電話後，會複製出一支「專門跟該客戶端通話」的伺服器端電話機。
    int listenFd, acceptFd, connFd;
    socklen_t len; /* Size of socket address buffer */
    void *addr; /* Buffer for socket address */
    char addrStr[IS_ADDR_STR_LEN];

    if (argc != 2 || strcmp(argv[1], "--help") == 0)
        usageErr("%s service\n", argv[0]);

    // 2. 程式執行流程拆解
    // 步驟一：伺服器啟動監聽
    // 動作：伺服器在指定的連接埠（由參數 argv[1] 輸入，例如 8080）開始監聽。
    // 隱含機制：這裡會自動幫伺服器做 bind() 與 listen()。同時，len 會被自動填入該網路 
    //          Domain（IPv4 或 IPv6）的地址結構長度。
    listenFd = inetListen(argv[1], 5, &len);
    if (listenFd == -1)
        errExit("inetListen");

    // 步驟二：客戶端主動連線（自己連自己）
    // 動作：客戶端主動去連線本機（第一個參數為 NULL 代表本地迴圈 localhost 或 127.0.0.1），
    //      目標埠口也是 argv[1]。
    // 隱含機制（隱式綁定）：因為客戶端沒有呼叫 bind()，所以核心會在這裡偷偷發配一個「隨機的臨時
    //                   埠口（Ephemeral Port）」給 connFd。
    connFd = inetConnect(NULL, argv[1], SOCK_STREAM);
    if (connFd == -1)
        errExit("inetConnect");

    // 步驟三：伺服器接受連線
    // 動作：監聽中的總機 listenFd 收到剛剛 connFd 撥進來的電話，按下接聽鍵，並建立一支專門通話
    //      用的 acceptFd。
    // 注意：這裡的 accept 後兩個參數填 NULL，代表伺服器在接聽時「故意不去看是誰連進來」，打算後
    //      面再用 getpeername() 來反查。
    acceptFd = accept(listenFd, NULL, NULL);
    if (acceptFd == -1)
        errExit("accept");

    // 步驟四：動態配置記憶體
    // 根據步驟一拿到的地址長度（len），動態挖一塊記憶體（addr 緩衝區），準備用來存放接下來查詢到
    // 的網路位址。
    addr = malloc(len);
    if (addr == NULL)
        errExit("malloc");

    // 3. 核心實驗：四大查詢結果預測
    // (1). 查詢客戶端(connFd)「自己」的地址；
    //      輸出預測：127.0.0.1:45678 （客戶端自己的臨時 Port）
    if (getsockname(connFd, addr, &len) == -1)
        errExit("getsockname");
    printf("getsockname(connFd): %s\n", inetAddressStr(addr, len, addrStr, IS_ADDR_STR_LEN));
    // (2). 查詢伺服器端專線(acceptFd)「自己」的地址；
    //      輸出預測：127.0.0.1:8080  （伺服器本機綁定的 Port）
    if (getsockname(acceptFd, addr, &len) == -1)
        errExit("getsockname");
    printf("getsockname(acceptFd): %s\n", inetAddressStr(addr, len, addrStr, IS_ADDR_STR_LEN));
    // (3). 查詢客戶端(connFd)「連線彼端」的地址；
    //      輸出預測：127.0.0.1:8080  （對客戶端來說，彼端就是伺服器）
    if (getpeername(connFd, addr, &len) == -1)
        errExit("getpeername");
    printf("getpeername(connFd): %s\n", inetAddressStr(addr, len, addrStr, IS_ADDR_STR_LEN));
    // (4). 查詢伺服器端專線(acceptFd)「連線彼端」的地址；
    //      輸出預測：127.0.0.1:45678 （對伺服器專線來說，彼端就是客戶端）
    if (getpeername(acceptFd, addr, &len) == -1)
        errExit("getpeername");
    printf("getpeername(acceptFd): %s\n", inetAddressStr(addr, len, addrStr, IS_ADDR_STR_LEN));

    sleep(30); /* Give us time to run netstat(8) */

    exit(EXIT_SUCCESS);
}
