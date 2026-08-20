/**
 * (進階的)實作malloc()與free()。
 * Compile Command : 
  gcc e7-2.c \
  -I/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
  -L/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib \
  -ltlpi -o e7-2_arm

  gcc e7-2.c \
  -DUSE_MYLIB_INTEL \
  -I/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib-intel \
  -L/home/earvin/workspaces/GithubProjects/BOOKS_STUDY/TLPI/tlpi-book/mylib-intel \
  -ltlpi -o e7-2_intel

 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h> // 提供 sbrk()
#include <stddef.h> // 提供 size_t
#include <stdbool.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif


// 記憶體區塊標頭
typedef struct Block {
    size_t size;         // 區塊大小（不含 Header 本身）
    bool is_free;        // 是否已釋放（true 表示可用）
    struct Block *next;  // 指向下一個區塊的指標
} Block;

#define BLOCK_SIZE sizeof(Block)

static Block *head = NULL; // 鏈結串列的頭指標

// 尋找足夠大的空閒區塊
Block *find_free_block(Block **last, size_t size) {
    Block *current = head;
    while (current && !(current->is_free && current->size >= size)) {
        *last = current;
        current = current->next;
    }
    return current;
}

// 向作業系統請求新空間
Block *request_space(Block *last, size_t size) {
    Block *block = sbrk(0); // 取得當前 break 位置
    void *request = sbrk(size + BLOCK_SIZE);
    
    if (request == (void*)-1) {
        return NULL; // 記憶體配置失敗
    }

    if (last) { // 將新區塊掛入串列尾端
        last->next = block;
    }
    
    block->size = size;
    block->is_free = false;
    block->next = NULL;
    return block;
}

void *my_malloc(size_t size) {
    if (size <= 0) return NULL;

    Block *block;

    if (!head) { // 第一位呼叫者，初始化鏈結串列
        block = request_space(NULL, size);
        if (!block) return NULL;
        head = block;
    } else {
        Block *last = head;
        block = find_free_block(&last, size);
        if (!block) { // 找不到合適的空閒區塊，向 OS 申請
            block = request_space(last, size);
            if (!block) return NULL;
        } else { // 找到可用區塊，標記為已使用
            block->is_free = false;
        }
    }

    // 回傳跳過 Header 後的實際資料起始位址
    return (void*)(block + 1);
}

// 取得標頭位址
Block *get_block_ptr(void *ptr) {
    return (Block*)ptr - 1;
}

void my_free(void *ptr) {
    if (!ptr) return;

    Block *block = get_block_ptr(ptr);
    block->is_free = true;
    
    // 註：工業級實作在此處會進行「區塊合併（Coalescing）」，
    // 將相鄰的 free 區塊融合成更大區塊，防止記憶體碎片化。
}

int main(int argc, char *argv[])
{
// 範例 1：配置整數陣列
    int *arr = (int *)my_malloc(5 * sizeof(int));
    if (!arr) {
        printf("陣列記憶體配置失敗！\n");
        return 1;
    }

    // 寫入並讀取資料
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    // 釋放整數陣列記憶體
    my_free(arr);
    printf("--> 整數陣列記憶體已釋放\n\n");

    // 範例 2：配置字串空間
    char *str = (char *)my_malloc(50 * sizeof(char));
    if (!str) {
        printf("字串記憶體配置失敗！\n");
        return 1;
    }

    // 複製字串並印出
    strcpy(str, "Hello, Custom Memory Allocator!");
    printf("字串內容: %s\n", str);

    // 釋放字串記憶體
    my_free(str);
    printf("--> 字串記憶體已釋放\n");

    return 0;
}
