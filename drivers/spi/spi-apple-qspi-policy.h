/* SPDX-License-Identifier: GPL-2.0 */
#ifndef APPLE_QSPI_POLICY_H
#define APPLE_QSPI_POLICY_H

#define APPLE_QSPI_NVRAM_START 0x700000ULL
#define APPLE_QSPI_NVRAM_END   0x800000ULL
#define APPLE_QSPI_PAGE_SIZE   256U
#define APPLE_QSPI_ERASE_SIZE  4096U

/* Subtraction avoids overflow; addr is not truncated to wire width first. */
static inline int apple_qspi_range_allowed(unsigned long long addr, unsigned int len)
{
	return len && addr >= APPLE_QSPI_NVRAM_START &&
		addr < APPLE_QSPI_NVRAM_END && len <= APPLE_QSPI_NVRAM_END - addr;
}

static inline int apple_qspi_mutation_allowed(unsigned int opcode,
		unsigned int addr_bytes, unsigned long long addr,
		unsigned int len, int data_out)
{
	switch (opcode) {
	case 0x06: /* Virtual WREN, never sent independently. */
	case 0x04: /* Virtual WRDI. */
		return !addr_bytes && !len && !data_out;
	case 0x02: /* PP: prevent page wrap, even if SPI-NOR gets geometry wrong. */
		return addr_bytes == 3 && data_out && len <= APPLE_QSPI_PAGE_SIZE &&
			apple_qspi_range_allowed(addr, len) &&
			len <= APPLE_QSPI_PAGE_SIZE - (addr & (APPLE_QSPI_PAGE_SIZE - 1));
	case 0x20: /* SE: hard-code the entire affected 4 KiB range. */
		return addr_bytes == 3 && !len && !data_out &&
			!(addr & (APPLE_QSPI_ERASE_SIZE - 1)) &&
			apple_qspi_range_allowed(addr, APPLE_QSPI_ERASE_SIZE);
	default:
		return 0;
	}
}

/*
 * T8140 boot NOR: may a request that does not read data reach the flash?
 * Write-enable control is admitted. A page program, or a 4, 32 or 64 KiB
 * erase, is admitted only when every byte it can change lies inside
 * [start, end), the nvram partition. A program counts the whole 4 KiB
 * blocks it touches, which covers a write that wraps within its page. Status
 * and configuration register writes, chip erase, address-mode, reset and OTP
 * commands are refused. The address must fit in addr_bytes: the controller
 * sends only those bytes, so a wider value would alias a lower address.
 */
static inline int apple_qspi_window_allowed(unsigned int opcode,
		unsigned int addr_bytes, unsigned long long addr,
		unsigned int len, int data_out,
		unsigned long long start, unsigned long long end)
{
	unsigned long long block, first, last;

	switch (opcode) {
	case 0x06: /* WREN */
	case 0x04: /* WRDI */
		return !addr_bytes && !len && !data_out;
	case 0x02: /* PP */
	case 0x12: /* PP_4B */
		if (!data_out || !len || len > APPLE_QSPI_ERASE_SIZE)
			return 0;
		block = APPLE_QSPI_ERASE_SIZE;
		break;
	case 0x20: /* BE_4K */
	case 0x21: /* BE_4K_4B */
		block = 0x1000;
		break;
	case 0x52: /* BE_32K */
	case 0x5c: /* BE_32K_4B */
		block = 0x8000;
		break;
	case 0xd8: /* SE */
	case 0xdc: /* SE_4B */
		block = 0x10000;
		break;
	default:
		return 0;
	}
	if (opcode != 0x02 && opcode != 0x12 && (len || data_out))
		return 0;
	if ((addr_bytes != 3 && addr_bytes != 4) || addr >> (8 * addr_bytes))
		return 0;

	/* [first, last) are the blocks the flash can change. */
	first = addr & ~(block - 1);
	last = ((addr + (len ? len : 1) - 1) | (block - 1)) + 1;
	return last <= 1ULL << (8 * addr_bytes) &&
		start <= first && last <= end;
}

/* Protection and activity bits must permit a write; JEDEC ID is checked separately. */
static inline int apple_qspi_status_allows_write(unsigned int sr1,
		unsigned int sr2, unsigned int sr3)
{
	/* BUSY/WEL, BP[2:0]; CMP/SUS; WPS. SRP, TB, SEC, QE, LB stay unchanged. */
	return !(sr1 & 0x1f) && !(sr2 & 0xc0) && !(sr3 & 0x04);
}
#endif
