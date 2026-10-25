// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Copyright 2026 Aurora Silicon */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include "afkv2-codec.h"
#define MAX_PENDING_CMDS 16
#define SZ_1M 1048576U
#define EACCES 13
#define spin_lock_irqsave(lock, flags) \
	do {                           \
		(void)(lock);          \
		(flags) = 0;           \
	} while (0)
#define spin_unlock_irqrestore(lock, flags) \
	do {                                \
		(void)(lock);               \
		(void)(flags);              \
	} while (0)
#define dma_rmb() \
	do {      \
	} while (0)
#define dma_wmb() \
	do {      \
	} while (0)
#define dev_warn_ratelimited(...) \
	do {                      \
	} while (0)
static int test_bit(unsigned int bit, const unsigned long *map)
{
	return !!(*map & (1UL << bit));
}

struct epic_cmd_info {
	bool done;
	void *rxbuf, *txbuf;
	u64 rxbuf_dma, txbuf_dma;
	size_t rxlen, txlen;
};

struct apple_dcp_afkep {
	bool wire_v2;
};

struct apple_epic_service {
	int lock;
	struct apple_dcp_afkep *ep;
	unsigned long cmd_map[1];
	struct epic_cmd_info cmds[MAX_PENDING_CMDS];
};
