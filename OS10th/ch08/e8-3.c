/*
 * e8-3.c
 * 實作銀行家演算法
 *
 * Compile Command : gcc e8-3.c -o e8-3
 * Created on: 2019年12月19日
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define M 5 // THREAD_COUNT
#define N 4 // RESOURCE_COUNT

int main(int argc, char *argv[])
{
    printf("This is a placeholder for the Banker's Algorithm implementation.\n");

    // process_count = 5, resource_count = 4
    int allocation[M][N] = {
        {0, 0, 1, 2},
        {1, 0, 0, 0},
        {1, 3, 5, 4},
        {0, 6, 3, 2},
        {0, 0, 1, 4}};

    int request[M][N] = {
        {0, 0, 1, 0},
        {1, 7, 5, 0},
        {2, 3, 5, 6},
        {0, 6, 5, 2},
        {0, 6, 5, 6}};

    // 目前各類資源(4類)可用數量
    int available[N] = {1, 8, 5, 4};

    // Thread counts ; value 0 = false, 1  = true
    int finish[M] = {FALSE, FALSE, FALSE, FALSE, FALSE};

    int work[N] = {0, 0, 0, 0};
    
    /*
     *  step1 : 先建立一個資源分配表，包含 allocation、request、available
     *  step2 : 實作安全性檢查函式，判斷目前的資源分配狀態是否安全
     *  step3 : 實作資源請求函式，判斷是否可以滿足某個行程的資源請求
     */
    memcpy(work, available, sizeof(available));

    for (int i = 0; i < M; i++) {
        printf("Process P%d: Allocation = {", i);
        for (int j = 0; j < N; j++) {
            // i = thread, j = resource
            printf("%d", allocation[i][j]);
            if (j < N - 1)
                printf(", ");
        }
        printf("}, Request = {");
        for (int j = 0; j < N; j++) {
            printf("%d", request[i][j]);
            if (j < 3)
                printf(", ");
        }
        printf("}, work = {");
        for (int j = 0; j < N; j++) {
            printf("%d", work[j]);
            if (j < 3)
                printf(", ");
        }
        printf("}\n");

        // Banker's Algorithm: Check if the process can be allocated resources
        if (finish[i] == FALSE) {
            int can_allocate = TRUE;
            for (int j = 0; j < N; j++) {
                if (request[i][j] > work[j]) {
                    can_allocate = FALSE;
                    break;
                }
            }

            if (can_allocate) {
                printf("Process P%d can be allocated resources.\n", i);
                for (int j = 0; j < N; j++) {
                    work[j] += allocation[i][j];
                }
                finish[i] = TRUE;
            } else {
                printf("Process P%d cannot be allocated resources.\n", i);
                return FALSE;   // unsafed, return
            }
        }
    }

    printf("All processes can be allocated resources safely.\n");

    exit(EXIT_SUCCESS);
}