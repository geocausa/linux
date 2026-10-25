// SPDX-License-Identifier: GPL-2.0
/*
 * KUnit tests for the T8140 boot NOR write window of the Apple QSPIMC driver.
 */

#include <kunit/test.h>

#include "spi-apple-qspi-policy.h"

/* The J700 nvram partition. */
#define NV_START	0x24000ULL
#define NV_END		0x1f4000ULL

static int nv(unsigned int opcode, unsigned int addr_bytes,
	      unsigned long long addr, unsigned int len, int data_out)
{
	return apple_qspi_window_allowed(opcode, addr_bytes, addr, len,
					 data_out, NV_START, NV_END);
}

static void qspi_window_write_enable(struct kunit *test)
{
	KUNIT_EXPECT_TRUE(test, nv(0x06, 0, 0, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0x04, 0, 0, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x06, 3, NV_START, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x06, 0, 0, 1, 1));
	/* Write-enable control needs no window. */
	KUNIT_EXPECT_TRUE(test, apple_qspi_window_allowed(0x06, 0, 0, 0, 0, 0, 0));
}

static void qspi_window_program(struct kunit *test)
{
	KUNIT_EXPECT_TRUE(test, nv(0x02, 3, NV_START, 256, 1));
	KUNIT_EXPECT_TRUE(test, nv(0x02, 3, NV_START + 0x1234, 1, 1));
	KUNIT_EXPECT_TRUE(test, nv(0x02, 3, NV_END - 256, 256, 1));
	KUNIT_EXPECT_TRUE(test, nv(0x12, 4, NV_START, 256, 1));
	/* Any byte outside, or a wrap within a page outside, is refused. */
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_START - 256, 256, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_START - 1, 2, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_END, 1, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_END - 1, 2, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, 0, 256, 1));
	/* A program needs data out, and at most one 4 KiB block's worth. */
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_START, 0, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_START, 256, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, NV_START, 4097, 1));
}

static void qspi_window_erase(struct kunit *test)
{
	KUNIT_EXPECT_TRUE(test, nv(0x20, 3, NV_START, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0x20, 3, NV_START + 0x123, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0x20, 3, NV_END - 1, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0x21, 4, NV_START, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 3, NV_START - 1, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 3, NV_END, 0, 0));
	/* 32 and 64 KiB erases cover the whole aligned block. */
	KUNIT_EXPECT_TRUE(test, nv(0x52, 3, 0x28000, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x52, 3, NV_START, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0xd8, 3, 0x30000, 0, 0));
	KUNIT_EXPECT_TRUE(test, nv(0xdc, 4, 0x1e0000, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0xd8, 3, NV_START, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0xd8, 3, 0x1f0000, 0, 0));
	/* An erase carries no data. */
	KUNIT_EXPECT_FALSE(test, nv(0x20, 3, NV_START, 1, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 3, NV_START, 0, 1));
}

static void qspi_window_refused_commands(struct kunit *test)
{
	static const u8 refused[] = {
		0x01, 0x31, 0x11,	/* status register writes */
		0x60, 0xc7,		/* chip erase */
		0xb7, 0xe9, 0xc5,	/* address mode, extended address */
		0x66, 0x99, 0xf0,	/* reset */
		0x42, 0x44, 0x98,	/* OTP program and erase, global unlock */
		0x32, 0x34, 0x38,	/* multi-line page programs */
		0x7a, 0x75, 0xb9, 0xab,	/* suspend, resume, power down */
	};
	int i;

	for (i = 0; i < ARRAY_SIZE(refused); i++) {
		KUNIT_EXPECT_FALSE(test, nv(refused[i], 0, 0, 0, 0));
		KUNIT_EXPECT_FALSE(test, nv(refused[i], 3, NV_START, 0, 0));
		KUNIT_EXPECT_FALSE(test, nv(refused[i], 3, NV_START, 1, 1));
	}
}

static void qspi_window_address_width(struct kunit *test)
{
	/* Three address bytes cannot carry an address above 16 MiB. */
	KUNIT_EXPECT_FALSE(test, nv(0x02, 3, (1ULL << 24) + NV_START, 1, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 3, (1ULL << 24) + NV_START, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x12, 4, (1ULL << 32) + NV_START, 1, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x02, 2, NV_START, 1, 1));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 0, NV_START, 0, 0));
	KUNIT_EXPECT_FALSE(test, nv(0x20, 5, NV_START, 0, 0));
	/* A window that reaches the top of 24-bit addressing. */
	KUNIT_EXPECT_TRUE(test, apple_qspi_window_allowed(0x20, 3, 0xfff000, 0, 0,
							  0xff0000, 1ULL << 24));
	/* A program that crosses 16 MiB cannot use three address bytes. */
	KUNIT_EXPECT_FALSE(test, apple_qspi_window_allowed(0x02, 3, 0xffffff, 2, 1,
							   0xff0000, 1ULL << 25));
	KUNIT_EXPECT_TRUE(test, apple_qspi_window_allowed(0x12, 4, 0xffffff, 2, 1,
							  0xff0000, 1ULL << 25));
}

static void qspi_window_empty(struct kunit *test)
{
	KUNIT_EXPECT_FALSE(test, apple_qspi_window_allowed(0x02, 3, 0, 1, 1, 0, 0));
	KUNIT_EXPECT_FALSE(test, apple_qspi_window_allowed(0x20, 3, 0, 0, 0, 0, 0));
	KUNIT_EXPECT_FALSE(test, apple_qspi_window_allowed(0x20, 3, NV_START, 0, 0,
							   NV_START, NV_START));
}

static struct kunit_case apple_qspi_window_cases[] = {
	KUNIT_CASE(qspi_window_write_enable),
	KUNIT_CASE(qspi_window_program),
	KUNIT_CASE(qspi_window_erase),
	KUNIT_CASE(qspi_window_refused_commands),
	KUNIT_CASE(qspi_window_address_width),
	KUNIT_CASE(qspi_window_empty),
	{}
};

static struct kunit_suite apple_qspi_window_suite = {
	.name = "apple-qspi-nvram-window",
	.test_cases = apple_qspi_window_cases,
};
kunit_test_suite(apple_qspi_window_suite);

MODULE_DESCRIPTION("KUnit tests for the Apple QSPIMC boot NOR write window");
MODULE_LICENSE("GPL");
