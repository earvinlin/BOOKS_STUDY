	.arch armv8-a
	.file	"setjmp_vars.c"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"Inside doJump(): nvar=%d rvar=%d vvar=%d\n"
	.align	3
.LC1:
	.string	"After longjmp(): nvar=%d rvar=%d vvar=%d\n"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB40:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp
	mov	w0, 333
	str	w0, [sp, 28]
	adrp	x0, .LANCHOR0
	add	x0, x0, :lo12:.LANCHOR0
	bl	_setjmp
	cbz	w0, .L5
	ldr	w4, [sp, 28]
	mov	w3, 222
	mov	w2, 111
	adrp	x1, .LC1
	add	x1, x1, :lo12:.LC1
	mov	w0, 2
	bl	__printf_chk
	mov	w0, 0
	bl	exit
.L5:
	mov	w0, 999
	str	w0, [sp, 28]
	ldr	w4, [sp, 28]
	mov	w3, 888
	mov	w2, 777
	adrp	x1, .LC0
	add	x1, x1, :lo12:.LC0
	mov	w0, 2
	bl	__printf_chk
	mov	w1, 1
	adrp	x0, .LANCHOR0
	add	x0, x0, :lo12:.LANCHOR0
	bl	__longjmp_chk
	.cfi_endproc
.LFE40:
	.size	main, .-main
	.bss
	.align	3
	.set	.LANCHOR0,. + 0
	.type	env, %object
	.size	env, 312
env:
	.zero	312
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
