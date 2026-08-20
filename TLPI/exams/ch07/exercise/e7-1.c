/**
 * 修改列表7-1的程式(free_and_sbrk.c)，在每次執行malloc()之後，印出目前的program break值。指
 * 定一個小的配置區塊來執行程式。這將能展示malloc()在每次呼明時不會用sbrk()調整program break ，
 * 而是定期分配大塊的記憶體，並每次傳回一小片記憶體給呼叫者。
 * command syntax :
 * ./e7-1_arm 1000 10240 1 1 999
 * ./e7-1_arm 1000 10240 1 500 1000
 */
#include <stdlib.h>
#include <errno.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

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
    freeStep = (argc > 3) ? getInt(argv[3], GN_GT_0, "step") : 1;
    freeMin = (argc > 4) ? getInt(argv[4], GN_GT_0, "min") : 1;
    freeMax = (argc > 5) ? getInt(argv[5], GN_GT_0, "max") : numAllocs;
    if (freeMax > numAllocs)
        cmdLineErr("free-max > num-allocs\n");

    // sbrk(0)的意思是：向作業系統查詢目前 Program Break(Heap 的邊界)所在的位置，但
    // 不對 Program Break 做任何移動。
    printf("Initial program break: %10p\n", sbrk(0));
    printf("Allocating %d*%d bytes\n", numAllocs, blockSize);

    for (j = 0; j < numAllocs; j++) {
        ptr[j] = malloc(blockSize);
        if (ptr[j] == NULL)
            errExit("malloc");
        
        // 印出第幾次分配、指標位址，以及當前的 Program Break
        printf("malloc[%2d] at %10p | Program break: %10p\n", j, ptr[j], sbrk(0));            
    }

    printf("Program break is now: %10p\n", sbrk(0));
    printf("Freeing blocks from %d to %d in steps of %d\n",
        freeMin, freeMax, freeStep);
    for (j = freeMin - 1; j < freeMax; j += freeStep)
        free(ptr[j]);
    printf("After free(), program break is: %10p\n", sbrk(0));
    
    exit(EXIT_SUCCESS);
}
