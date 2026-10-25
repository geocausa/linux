// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Included by macsmc.c; the command and SRAM transports are stubbed. */
#include <kunit/test.h>

struct macsmc_test_context {
	struct apple_smc smc;
	u8 shmem[SMC_MAX_SIZE];
	u32 data;
	int result;
	unsigned int calls;
	bool require_sram_lock;
};

static int macsmc_test_command(struct apple_smc *smc, u64 cmd, u64 arg,
			       u64 size, u64 wsize, u32 *ret_data)
{
	struct macsmc_test_context *ctx = container_of(smc, struct macsmc_test_context, smc);

	KUNIT_EXPECT_TRUE(kunit_get_current_test(), mutex_is_locked(&smc->mutex));
	ctx->calls++;
	if (ret_data && ctx->result >= 0)
		*ret_data = ctx->data;
	return ctx->result;
}

static void macsmc_test_read_sram(struct apple_smc *smc, void *buf, size_t size)
{
	struct macsmc_test_context *ctx = container_of(smc, struct macsmc_test_context, smc);

	if (ctx->require_sram_lock)
		KUNIT_EXPECT_TRUE(kunit_get_current_test(), mutex_is_locked(&smc->mutex));
	memcpy(buf, ctx->shmem, size);
}

static int macsmc_test_init(struct kunit *test)
{
	struct macsmc_test_context *ctx;

	ctx = kunit_kzalloc(test, sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;
	mutex_init(&ctx->smc.mutex);
	ctx->smc.shmem.iomem = (void __iomem *)ctx->shmem;
	ctx->smc.key_count = 3;
	test->priv = ctx;
	kunit_activate_static_stub(test, apple_smc_cmd_locked, macsmc_test_command);
	kunit_activate_static_stub(test, apple_smc_read_sram, macsmc_test_read_sram);
	return 0;
}

static void macsmc_value_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	u8 byte = 0;
	u32 word = 0;

	ctx->data = 0x12345678;
	ctx->result = sizeof(word);
	KUNIT_EXPECT_EQ(test, apple_smc_read_u32(&ctx->smc, SMC_KEY(TEST), &word), 0);
	KUNIT_EXPECT_EQ(test, word, ctx->data);
	ctx->result = sizeof(byte);
	KUNIT_EXPECT_EQ(test, apple_smc_read_u8(&ctx->smc, SMC_KEY(TEST), &byte), 0);
	KUNIT_EXPECT_EQ(test, byte, 0x78);
	ctx->result = -ETIMEDOUT;
	KUNIT_EXPECT_EQ(test, apple_smc_read_u8(&ctx->smc, SMC_KEY(TEST), &byte), -ETIMEDOUT);
	KUNIT_EXPECT_EQ(test, byte, 0x78);
}

static void macsmc_fixed_value_short_reply_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	u32 word = 0xa5a5a5a5;
	u8 byte = 0xa5;

	ctx->data = 0x12345678;
	ctx->result = 2;
	KUNIT_EXPECT_EQ(test, apple_smc_read_u32(&ctx->smc, SMC_KEY(TEST), &word), -EIO);
	KUNIT_EXPECT_EQ(test, word, 0U);
	word = 0xa5a5a5a5;
	KUNIT_EXPECT_EQ(test,
			apple_smc_rw_u32(&ctx->smc, SMC_KEY(TEST), 0, &word), -EIO);
	KUNIT_EXPECT_EQ(test, word, 0U);
	ctx->result = 0;
	KUNIT_EXPECT_EQ(test, apple_smc_read_u8(&ctx->smc, SMC_KEY(TEST), &byte), -EIO);
	KUNIT_EXPECT_EQ(test, byte, 0);
}

static void macsmc_request_bounds_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	u8 byte;

	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), &byte,
					     SMC_MAX_SIZE + 1), -EINVAL);
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), &byte, SIZE_MAX),
			-EINVAL);
	KUNIT_EXPECT_EQ(test, apple_smc_write(&ctx->smc, SMC_KEY(TEST), &byte, SIZE_MAX),
			-EINVAL);
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), &byte, 0), -EINVAL);
	KUNIT_EXPECT_EQ(test, ctx->calls, 0U);
}

static void macsmc_sram_value_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	u8 value[SMC_MAX_SIZE + 2];
	unsigned int i;

	for (i = 0; i < sizeof(ctx->shmem); i++)
		ctx->shmem[i] = i;
	memset(value, 0xa5, sizeof(value));
	ctx->result = SMC_MAX_SIZE;
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), value + 1,
					     SMC_MAX_SIZE), SMC_MAX_SIZE);
	KUNIT_EXPECT_MEMEQ(test, value + 1, ctx->shmem, SMC_MAX_SIZE);
	KUNIT_EXPECT_EQ(test, value[0], 0xa5);
	KUNIT_EXPECT_EQ(test, value[SMC_MAX_SIZE + 1], 0xa5);
}

static void macsmc_key_info_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	struct apple_smc_key_info info = {};
	const u8 wire[] = { 255, 'u', 'i', '3', '2', APPLE_SMC_READABLE };

	memcpy(ctx->shmem, wire, sizeof(wire));
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_info(&ctx->smc, SMC_KEY(TEST), &info), 0);
	KUNIT_EXPECT_EQ(test, info.size, 255);
	KUNIT_EXPECT_EQ(test, info.type_code, SMC_KEY(ui32));
	KUNIT_EXPECT_EQ(test, info.flags, APPLE_SMC_READABLE);
	ctx->result = -EIO;
	memset(ctx->shmem, 0, sizeof(wire));
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_info(&ctx->smc, SMC_KEY(TEST), &info), -EIO);
	KUNIT_EXPECT_EQ(test, info.size, 255);
}

static void macsmc_short_value_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	u8 value[4];

	memset(value, 0xa5, sizeof(value));
	ctx->data = 0x12345678;
	ctx->result = 0;
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), value, 4), 0);
	KUNIT_EXPECT_MEMEQ(test, value, ((u8[]){ 0xa5, 0xa5, 0xa5, 0xa5 }), 4);
	ctx->result = 2;
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), value, 4), 2);
	KUNIT_EXPECT_MEMEQ(test, value, ((u8[]){ 0x78, 0x56, 0xa5, 0xa5 }), 4);
	memset(value, 0xa5, sizeof(value));
	ctx->result = 5;
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), value, 4), -EIO);
	KUNIT_EXPECT_MEMEQ(test, value, ((u8[]){ 0xa5, 0xa5, 0xa5, 0xa5 }), 4);
	ctx->result = 7;
	memset(ctx->shmem, 0x3c, sizeof(ctx->shmem));
	memset(ctx->shmem + 16, 0xa5, 8);
	KUNIT_EXPECT_EQ(test, apple_smc_read(&ctx->smc, SMC_KEY(TEST), ctx->shmem + 16, 8), 7);
	KUNIT_EXPECT_EQ(test, ctx->shmem[23], 0xa5);
}

static void macsmc_key_error_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	smc_key key = SMC_KEY(TEST);

	ctx->result = -EIO;
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_by_index(&ctx->smc, 0, &key), -EIO);
	KUNIT_EXPECT_EQ(test, key, SMC_KEY(TEST));
	ctx->result = 0;
	ctx->data = swab32(SMC_KEY(TEST));
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_by_index(&ctx->smc, 2, &key), 0);
	KUNIT_EXPECT_EQ(test, key, SMC_KEY(TEST));
}

static void macsmc_key_index_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	smc_key key = SMC_KEY(TEST);
	int index = -1;

	KUNIT_EXPECT_EQ(test,
			apple_smc_get_key_by_index(&ctx->smc, READ_ONCE(index), &key),
			-EINVAL);
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_by_index(&ctx->smc, 3, &key), -EINVAL);
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_by_index(&ctx->smc, INT_MAX, &key), -EINVAL);
	KUNIT_EXPECT_EQ(test, ctx->calls, 0U);
	KUNIT_EXPECT_EQ(test, key, SMC_KEY(TEST));
}

static void macsmc_key_info_lock_test(struct kunit *test)
{
	struct macsmc_test_context *ctx = test->priv;
	struct apple_smc_key_info info;

	ctx->require_sram_lock = true;
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_info(&ctx->smc, SMC_KEY(TEST), &info), 0);
	KUNIT_EXPECT_FALSE(test, mutex_is_locked(&ctx->smc.mutex));
	KUNIT_EXPECT_EQ(test, apple_smc_get_key_info(&ctx->smc, SMC_KEY(TEST), NULL), 0);
	KUNIT_EXPECT_FALSE(test, mutex_is_locked(&ctx->smc.mutex));
}

static struct kunit_case macsmc_parser_cases[] = {
	KUNIT_CASE(macsmc_key_info_lock_test),
	KUNIT_CASE(macsmc_key_index_test),
	KUNIT_CASE(macsmc_key_error_test),
	KUNIT_CASE(macsmc_short_value_test),
	KUNIT_CASE(macsmc_fixed_value_short_reply_test),
	KUNIT_CASE(macsmc_value_test),
	KUNIT_CASE(macsmc_request_bounds_test),
	KUNIT_CASE(macsmc_sram_value_test),
	KUNIT_CASE(macsmc_key_info_test),
	{}
};

static struct kunit_suite macsmc_parser_suite = {
	.name = "macsmc-parser",
	.init = macsmc_test_init,
	.test_cases = macsmc_parser_cases,
};

kunit_test_suite(macsmc_parser_suite);
