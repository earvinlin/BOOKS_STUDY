
/*
    請設計一個程式，以init為根節點，畫出一棵樹來呈現系統上全部行程的父子關係，程式要顯示行程 ID與執行指令。
    程式輸出應該類似pstree（l）的輸出，但不用很詳細。系統上每個行程都可以透過檢測/proc/PID/status檔案內
    容，從包含 ppid：的那一行找出父行程。不過要小心一個問題，就是行程父行程（以及其/proc/PID 目錄）可能會
    在掃描全部的/proc/PID 日錄期間消失。
    Compile Cmd : gcc e12-2.c -o e12-2_arm
    -- if needs to debug... --
    gcc -g e12-2.c -o e12-2_arm
    sudo apt install valgrind
    valgrind ./e12-2_arm
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>

#define MAX_PROC 32768

typedef struct ProcessNode {
    int pid;
    int ppid;
    char name[256];
    int child_count;
    struct ProcessNode *children[128];
} ProcessNode;

ProcessNode *nodes[MAX_PROC] = {NULL};

// 遞迴印出行程樹
void print_tree(ProcessNode *node, int depth) {
    if (!node) return;
    
    for (int i = 0; i < depth; i++) printf("  ");
    printf("|- %s(%d)\n", node->name, node->pid);

    for (int i = 0; i < node->child_count; i++) {
        print_tree(node->children[i], depth + 1);
    }
}

int main() {
    DIR *dir = opendir("/proc");
    if (!dir) return 1;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // 判斷目錄名稱是否全為數字 (PID)
        if (isdigit(entry->d_name[0])) {
            int pid = atoi(entry->d_name);
            char path[256];
            snprintf(path, sizeof(path), "/proc/%d/status", pid);

            FILE *fp = fopen(path, "r");
            if (!fp) continue; // 行程可能在掃描中消失，直接跳過

            char line[256], name[256] = "unknown";
            int ppid = 0;

            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "Name:", 5) == 0) {
                    sscanf(line + 5, "%s", name);
                } else if (strncmp(line, "PPid:", 5) == 0) {
                    sscanf(line + 5, "%d", &ppid);
                }
            }
            fclose(fp);

            // 建立節點
            ProcessNode *node = malloc(sizeof(ProcessNode));
            node->pid = pid;
            node->ppid = ppid;
            strcpy(node->name, name);
            node->child_count = 0;
            nodes[pid] = node;
        }
    }
    closedir(dir);

    // 建立父子關係樹
    for (int i = 1; i < MAX_PROC; i++) {
        if (nodes[i] && nodes[i]->ppid < MAX_PROC && nodes[nodes[i]->ppid]) {
            ProcessNode *parent = nodes[nodes[i]->ppid];
            if (parent->pid != nodes[i]->pid) { // 避免根節點自環
                parent->children[parent->child_count++] = nodes[i];
            }
        }
    }

    // 從 PID 1 (init/systemd) 開始印出
    if (nodes[1]) {
        print_tree(nodes[1], 0);
    }

    return 0;
}