/**
 * #define _BSD_SOURCE <-- 已被棄用
 */
#include "../../tlpi-book/mylib/tlpi_hdr.h"
#define MAX_ALLOCS 1000000

int main(int argc, char *argv[])
{
    char *ptr[MAX_ALLOCS];
    int freeStep, freeMin, freeMax, blockSize, numAllocs, j;
    printf("\n");

    if (argc < 3 || strcmp(argv[1], "--help") == 0)
        usageErr("%s num-allocs block-size [step [min [max]]]\n", argv[0]);

    numAllocs = getInt(argv[1], GN_GT_0, "num-allocs");
    if (numAllocs > MAX_ALLOCS)
        cmdLineErr("num-allocs > %d\n", MAX_ALLOCS);

    blockSize = getInt(argv[2], GN_GT_0 | GN_ANY_BASE, "block-size");
    // 讓使用者可以選擇是否提供「釋放記憶體的間隔（step）」，若未提供則使用預設值 1。
    freeStep = (argc > 3) ? getInt(argv[3], GN_GT_0, "step") : 1;
    // 讓使用者可以選擇是否指定「從第幾個記憶體區塊開始釋放」，若未指定則預設從第 1 個開始。
    freeMin = (argc > 4) ? getInt(argv[4], GN_GT_0, "min") : 1;
    // 讓使用者可以選擇是否指定「釋放記憶體的最大區塊編號」，若未指定則預設釋放到全部（numAllocs）。
    freeMax = (argc > 5) ? getInt(argv[5], GN_GT_0, "max") : numAllocs;

    if (freeMax > numAllocs)
        cmdLineErr("free-max > num-allocs\n");

    printf("Initial program break: %10p\n", sbrk(0));
    printf("Allocating %d*%d bytes\n", numAllocs, blockSize);

    for (j = 0; j < numAllocs; j++) {
        ptr[j] = malloc(blockSize);
        if (ptr[j] == NULL)
            errExit("malloc");
    }

    printf("Program break is now: %10p\n", sbrk(0));
    printf("Freeing blocks from %d to %d in steps of %d\n",
    
    freeMin, freeMax, freeStep);
    for (j = freeMin - 1; j < freeMax; j += freeStep)
        free(ptr[j]);

    printf("After free(), program break is: %10p\n", sbrk(0));

    exit(EXIT_SUCCESS);
}