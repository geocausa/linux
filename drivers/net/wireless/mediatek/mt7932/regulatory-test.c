// SPDX-License-Identifier: GPL-2.0-only
/*
 * KUnit tests for the regulatory snapshot logic, the power package
 * parser and channel filter, and the country package lookup with its
 * world fallback. The packages are local files, so the parser treats
 * them as untrusted.
 */
#include <kunit/test.h>
#include <linux/prandom.h>

#include "mt7932.h"

/* A world-domain snapshot: channels 1, 6, 36 and 149, the 5 GHz ones NO_IR. */
static void reg_build(struct mt7932_reg_snapshot *reg)
{
	static const u16 channels[] = {1, 6, 36, 149};
	unsigned int i;

	memset(reg, 0, sizeof(*reg));
	memcpy(reg->alpha2, "00", 2);
	memcpy(reg->domain, "XZ", 2);
	reg->domain[4] = 1;
	reg->domain[5] = 2;
	reg->domain[8] = 2;
	reg->domain[9] = 2;
	for (i = 0; i < ARRAY_SIZE(channels); i++) {
		put_unaligned_le16(channels[i], reg->domain + 12 + i * 8);
		put_unaligned_le32(channels[i] > 14 ? IEEE80211_CHAN_NO_IR : 0,
				   reg->domain + 16 + i * 8);
		reg->power[i] = 20;
	}
	reg->length = 12 + ARRAY_SIZE(channels) * 8;
}

static void reg_set_flags(struct mt7932_reg_snapshot *reg, unsigned int entry, u32 flags)
{
	put_unaligned_le32(flags, reg->domain + 16 + entry * 8);
}

/* A beacon hint lifting NO_IR only relaxes; nothing else does. */
static void mt7932_reg_relaxation_test(struct kunit *test)
{
	struct mt7932_reg_snapshot *prev, *next;

	prev = kunit_kzalloc(test, sizeof(*prev), GFP_KERNEL);
	next = kunit_kzalloc(test, sizeof(*next), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, prev);
	KUNIT_ASSERT_NOT_NULL(test, next);
	reg_build(prev);

	reg_build(next);
	reg_set_flags(next, 2, 0);
	KUNIT_EXPECT_TRUE(test, mt7932_reg_only_relaxes(prev, next));
	/* An identical snapshot is not treated as a change by the caller,
	 * but it is trivially within the applied domain.
	 */
	KUNIT_EXPECT_TRUE(test, mt7932_reg_only_relaxes(prev, prev));

	/* NO_IR added: a restriction. */
	reg_build(next);
	reg_set_flags(next, 0, IEEE80211_CHAN_NO_IR);
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	/* Radar or a bandwidth limit appears, or another flag goes away. */
	reg_build(next);
	reg_set_flags(next, 2, IEEE80211_CHAN_NO_IR | IEEE80211_CHAN_RADAR);
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_set_flags(prev, 3, IEEE80211_CHAN_NO_IR | IEEE80211_CHAN_NO_HT40);
	reg_build(next);
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_build(prev);
	/* A new country, power limit, channel or channel order. */
	reg_build(next);
	memcpy(next->alpha2, "AU", 2);
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_build(next);
	next->power[1] = 19;
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_build(next);
	put_unaligned_le16(40, next->domain + 12 + 3 * 8);
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_build(next);
	put_unaligned_le16(44, next->domain + 12 + 4 * 8);
	next->domain[9] = 3;
	next->length += 8;
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	/* A failed snapshot never relaxes, and a corrupt length is refused. */
	reg_build(next);
	next->error = -EOPNOTSUPP;
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
	reg_build(next);
	prev->length = sizeof(prev->domain) + 8;
	next->length = prev->length;
	KUNIT_EXPECT_FALSE(test, mt7932_reg_only_relaxes(prev, next));
}

#define PKG_MODE_SIZE 328
#define PKG_ENTRY_SIZE 122

/* Channels of the synthetic 2.4 and 5 GHz tables 1..9, in table order. */
static const u8 pkg_channels[] = {
	1, 2, 3, 4, 5, 6, 7, 8,
	9, 10, 11, 12, 13, 14,
	36, 40, 44, 48, 52, 56, 60, 64,
	100, 104, 108, 112, 116, 120, 124, 128,
	132, 136, 140, 144, 149, 153, 157, 161,
	38, 46, 54, 62, 102, 110, 118, 126,
	134, 142, 151, 159, 42, 58, 106, 122,
	138, 155, 50, 114, 130, 146, 163, 167,
	165, 169, 171, 173, 175,
};

static unsigned int pkg_count(unsigned int table, bool world)
{
	if (table == 2)
		return world ? 6 : 5;
	return table == 9 ? 5 : 8;
}

static size_t pkg_size(bool world)
{
	static const u16 others[] = {532, 144, 58, 265};
	size_t size = 16 + (world ? 2 : 3) * PKG_MODE_SIZE;
	unsigned int i;

	for (i = 1; i <= 9; i++)
		size += 44 + pkg_count(i, world) * PKG_ENTRY_SIZE;
	for (i = 0; i < ARRAY_SIZE(others); i++)
		size += others[i];
	return size;
}

/* Build a structurally valid package. Channel 14 is only in the world one. */
static u8 *pkg_build(struct kunit *test, const char *alpha2, size_t *size)
{
	static const u16 lengths[] = {532, 144, 58, 265};
	static const u8 types[] = {5, 1, 4, 2};
	bool world = !memcmp(alpha2, "XZ", 2);
	unsigned int i, j, modes = world ? 2 : 3, channel = 0;
	u8 *data, *p;

	*size = pkg_size(world);
	data = kunit_kzalloc(test, *size, GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, data);
	memcpy(data, "J7RP", 4);
	put_unaligned_le16(1, data + 4);
	memcpy(data + 6, alpha2, 2);
	data[8] = modes;
	data[9] = MT7932_POLICY_TABLES;
	put_unaligned_le32(*size, data + 12);
	p = data + 16;
	for (i = 0; i < modes; i++, p += PKG_MODE_SIZE)
		put_unaligned_le32(17 + i, p + 4);
	for (i = 0; i < MT7932_POLICY_TABLES; i++) {
		if (i >= 1 && i <= 9) {
			unsigned int count = pkg_count(i, world);

			p[0] = 3;
			p[1] = 0x0a;
			put_unaligned_le16(44 + count * PKG_ENTRY_SIZE, p + 2);
			p[4] = count;
			p[5] = i <= 2 ? 1 : 2;
			p[6] = i == 2 || i == 9;
			memcpy(p + 8, alpha2, 2);
			for (j = 0; j < count; j++) {
				u8 *entry = p + 44 + j * PKG_ENTRY_SIZE;

				/* A country package has no channel 14 entry. */
				if (!world && pkg_channels[channel] == 14)
					channel++;
				entry[0] = pkg_channels[channel++];
				memset(entry + 1, 30, PKG_ENTRY_SIZE - 1);
			}
			p += 44 + count * PKG_ENTRY_SIZE;
		} else {
			unsigned int k = i ? i - 9 : 0;

			p[1] = 2;
			put_unaligned_le16(lengths[k], p + 2);
			p[7] = types[k];
			if (i == 10)
				memcpy(p + 8, alpha2, 2);
			p += lengths[k];
		}
	}
	KUNIT_ASSERT_PTR_EQ(test, p, data + *size);
	return data;
}

/* The entry of @channel in a parsed package, or NULL. */
static u8 *pkg_entry(const struct mt7932_policy *policy, unsigned int channel)
{
	unsigned int i, j;

	for (i = 1; i <= 9; i++)
		for (j = 0; j < policy->table[i][4]; j++)
			if (policy->table[i][44 + j * PKG_ENTRY_SIZE] == channel)
				return (u8 *)policy->table[i] + 44 + j * PKG_ENTRY_SIZE;
	return NULL;
}

static void mt7932_policy_valid_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data;

	data = pkg_build(test, "AU", &size);
	KUNIT_EXPECT_EQ(test, size, 10447);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "AU"), 0);
	KUNIT_EXPECT_EQ(test, policy.modes, 3);
	KUNIT_EXPECT_PTR_EQ(test, policy.table[0], data + 16 + 3 * PKG_MODE_SIZE);
	KUNIT_EXPECT_EQ(test, policy.length[12], 265);
	KUNIT_EXPECT_TRUE(test, mt7932_policy_permits(&policy, 36));
	KUNIT_EXPECT_FALSE(test, mt7932_policy_permits(&policy, 14));

	data = pkg_build(test, "XZ", &size);
	KUNIT_EXPECT_EQ(test, size, 10241);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "XZ"), 0);
	KUNIT_EXPECT_EQ(test, policy.modes, 2);
	KUNIT_EXPECT_TRUE(test, mt7932_policy_permits(&policy, 14));
}

/* A package serves only the country it names, in the header and table 10. */
static void mt7932_policy_country_mismatch_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data;

	data = pkg_build(test, "AU", &size);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "XZ"), -EINVAL);
	/* A country package renamed to the world one is not the world one. */
	memcpy(data + 6, "XZ", 2);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "XZ"), -EINVAL);
}

/* Every prefix of a valid package, with a consistent header, is rejected. */
static void mt7932_policy_truncation_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size, cut;
	u8 *data, *copy;

	data = pkg_build(test, "PL", &size);
	for (cut = 0; cut < size; cut++) {
		/* An exact-size copy lets KASAN catch any read past the end. */
		copy = kmemdup(data, cut, GFP_KERNEL);
		KUNIT_ASSERT_TRUE(test, copy || !cut);
		if (cut >= 16)
			put_unaligned_le32(cut, copy + 12);
		KUNIT_EXPECT_EQ_MSG(test, mt7932_policy_parse(&policy, copy, cut, "PL"),
				    -EINVAL, "length %zu", cut);
		kfree(copy);
	}
}

static void mt7932_policy_header_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data, *longer;

	data = pkg_build(test, "PL", &size);
	/* Trailing bytes after the last table. */
	longer = kunit_kzalloc(test, size + 4, GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, longer);
	memcpy(longer, data, size);
	put_unaligned_le32(size + 4, longer + 12);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, longer, size + 4, "PL"), -EINVAL);

	data[4] = 2;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	data[4] = 1;
	data[8] = 4;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	data[8] = 3;
	data[9] = 12;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	data[9] = MT7932_POLICY_TABLES;
	put_unaligned_le32(size - 1, data + 12);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	put_unaligned_le32(size, data + 12);
	/* Mode command lengths are bounded by the 320-byte body. */
	put_unaligned_le32(321, data + 16 + 4);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	put_unaligned_le32(0, data + 16 + 4);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	put_unaligned_le32(17, data + 16 + 4);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), 0);
}

/* Table lengths and entry counts must agree with the table type. */
static void mt7932_policy_table_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data, *table;

	data = pkg_build(test, "PL", &size);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), 0);
	table = (u8 *)policy.table[3];
	/* An entry count that would run past the table. */
	table[4] = 9;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	table[4] = 8;
	/* A table length that claims the rest of the file. */
	put_unaligned_le16(0xffff, table + 2);
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
	put_unaligned_le16(44 + 8 * PKG_ENTRY_SIZE, table + 2);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), 0);
	table = (u8 *)policy.table[10];
	table[7] = 2;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), -EINVAL);
}

/* The world package's 2.4 GHz tables must list channels 1..14 in order. */
static void mt7932_policy_world_order_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data;

	data = pkg_build(test, "XZ", &size);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "XZ"), 0);
	pkg_entry(&policy, 9)[0] = 10;
	KUNIT_EXPECT_EQ(test, mt7932_policy_parse(&policy, data, size, "XZ"), -EINVAL);
}

static void mt7932_policy_permits_test(struct kunit *test)
{
	struct mt7932_policy policy;
	size_t size;
	u8 *data, *entry;

	data = pkg_build(test, "PL", &size);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), 0);
	entry = pkg_entry(&policy, 144);
	KUNIT_ASSERT_NOT_NULL(test, entry);
	memset(entry + 1, MT7932_POLICY_NOT_PERMITTED, PKG_ENTRY_SIZE - 1);
	KUNIT_EXPECT_FALSE(test, mt7932_policy_permits(&policy, 144));
	/* One rate with a limit is enough to permit the channel. */
	entry[PKG_ENTRY_SIZE - 1] = 8;
	KUNIT_EXPECT_TRUE(test, mt7932_policy_permits(&policy, 144));
	/* A channel the package does not list is not permitted. */
	KUNIT_EXPECT_FALSE(test, mt7932_policy_permits(&policy, 177));
}

static void mt7932_policy_filter_test(struct kunit *test)
{
	static const u16 channels[] = {1, 6, 13, 36, 144, 149, 165};
	struct mt7932_reg_snapshot *reg;
	struct mt7932_policy policy;
	unsigned int i;
	size_t size;
	u8 *data;

	reg = kunit_kzalloc(test, sizeof(*reg), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, reg);
	data = pkg_build(test, "PL", &size);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&policy, data, size, "PL"), 0);
	memset(pkg_entry(&policy, 144) + 1, MT7932_POLICY_NOT_PERMITTED,
	       PKG_ENTRY_SIZE - 1);
	memset(pkg_entry(&policy, 6) + 1, MT7932_POLICY_NOT_PERMITTED,
	       PKG_ENTRY_SIZE - 1);
	memcpy(reg->domain, "PL", 2);
	for (i = 0; i < ARRAY_SIZE(channels); i++) {
		put_unaligned_le16(channels[i], reg->domain + 12 + i * 8);
		put_unaligned_le32(i, reg->domain + 16 + i * 8);
	}
	reg->domain[8] = 3;
	reg->domain[9] = 4;
	reg->length = 12 + ARRAY_SIZE(channels) * 8;

	mt7932_policy_filter(reg, &policy);
	KUNIT_EXPECT_EQ(test, reg->length, 12 + 5 * 8);
	KUNIT_EXPECT_EQ(test, reg->domain[8], 2);
	KUNIT_EXPECT_EQ(test, reg->domain[9], 3);
	KUNIT_EXPECT_EQ(test, get_unaligned_le16(reg->domain + 12 + 8), 13);
	KUNIT_EXPECT_EQ(test, get_unaligned_le16(reg->domain + 12 + 3 * 8), 149);
	/* Each entry keeps its own channel flags. */
	KUNIT_EXPECT_EQ(test, get_unaligned_le32(reg->domain + 16 + 3 * 8), 5);
	/* The vacated tail is cleared. */
	KUNIT_EXPECT_TRUE(test, !memchr_inv(reg->domain + 12 + 5 * 8, 0, 2 * 8));
}

/* Random corruption must never be accepted in a way that reads out of bounds. */
static void mt7932_policy_corruption_test(struct kunit *test)
{
	struct rnd_state rnd;
	struct mt7932_policy policy;
	unsigned int round, flips, channel;
	size_t size;
	u8 *data, *copy;

	prandom_seed_state(&rnd, 7932);
	data = pkg_build(test, "AU", &size);
	copy = kunit_kzalloc(test, size, GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, copy);
	for (round = 0; round < 2000; round++) {
		memcpy(copy, data, size);
		for (flips = 1 + prandom_u32_state(&rnd) % 4; flips; flips--)
			copy[prandom_u32_state(&rnd) % size] ^= 1 << (prandom_u32_state(&rnd) % 8);
		if (mt7932_policy_parse(&policy, copy, size, "AU"))
			continue;
		for (channel = 0; channel < 256; channel++)
			mt7932_policy_permits(&policy, channel);
	}
}

#define FAKE_FILES 8
#define FAKE_LOG 16

/* An in-memory firmware directory that records every lookup. */
struct fake_fs {
	struct {
		const char *path;
		const u8 *data;
		size_t size;
	} file[FAKE_FILES];
	unsigned int files, held;
	char log[FAKE_LOG][MT7932_POLICY_PATH_MAX];
	unsigned int requests;
};

static int fake_get(void *ctx, const char *path, struct mt7932_policy_file *file)
{
	struct fake_fs *fs = ctx;
	unsigned int i;

	if (fs->requests < FAKE_LOG)
		strscpy(fs->log[fs->requests], path, MT7932_POLICY_PATH_MAX);
	fs->requests++;
	for (i = 0; i < fs->files; i++) {
		if (strcmp(fs->file[i].path, path))
			continue;
		file->handle = fs;
		file->data = fs->file[i].data;
		file->size = fs->file[i].size;
		fs->held++;
		return 0;
	}
	return -ENOENT;
}

static void fake_put(void *ctx, struct mt7932_policy_file *file)
{
	struct fake_fs *fs = ctx;

	fs->held--;
	file->handle = NULL;
}

static void fake_add(struct fake_fs *fs, const char *path, const u8 *data, size_t size)
{
	fs->file[fs->files].path = path;
	fs->file[fs->files].data = data;
	fs->file[fs->files].size = size;
	fs->files++;
}

/* A country snapshot as the notifier builds it: 1, 6, 13, 36 and 149. */
static void reg_build_country(struct mt7932_reg_snapshot *reg, const char *alpha2)
{
	static const u16 channels[] = {1, 6, 13, 36, 149};
	unsigned int i;

	memset(reg, 0, sizeof(*reg));
	memcpy(reg->alpha2, alpha2, 2);
	memcpy(reg->domain, alpha2, 2);
	if (!memcmp(alpha2, "00", 2))
		memcpy(reg->domain, "XZ", 2);
	reg->domain[4] = 1;
	reg->domain[5] = 2;
	reg->domain[8] = 3;
	reg->domain[9] = 2;
	for (i = 0; i < ARRAY_SIZE(channels); i++) {
		put_unaligned_le16(channels[i], reg->domain + 12 + i * 8);
		reg->power[i] = 20;
	}
	reg->length = 12 + ARRAY_SIZE(channels) * 8;
}

struct select_env {
	struct fake_fs fs;
	struct mt7932_policy policy;
	struct mt7932_policy_file file;
	struct mt7932_policy_choice choice;
	struct mt7932_reg_snapshot reg;
	struct mt7932_policy_loader loader;
};

static struct select_env *select_env(struct kunit *test)
{
	struct select_env *env = kunit_kzalloc(test, sizeof(*env), GFP_KERNEL);

	KUNIT_ASSERT_NOT_NULL(test, env);
	env->loader.get = fake_get;
	env->loader.put = fake_put;
	env->loader.ctx = &env->fs;
	return env;
}

static int select_run(struct select_env *env, const char *alpha2)
{
	if (env->file.handle)
		fake_put(&env->fs, &env->file);
	env->fs.requests = 0;
	reg_build_country(&env->reg, alpha2);
	return mt7932_policy_select(&env->loader, &env->reg, &env->policy, &env->file,
				    &env->choice);
}

/* Lookup order: policy/CC, then world; 00 only the world package. */
static void mt7932_policy_order_test(struct kunit *test)
{
	enum mt7932_policy_source sources[MT7932_POLICY_CANDIDATES];
	char path[MT7932_POLICY_PATH_MAX];
	struct select_env *env = select_env(test);

	KUNIT_ASSERT_EQ(test, mt7932_policy_candidates("AU", sources), 2);
	KUNIT_EXPECT_EQ(test, sources[0], MT7932_POLICY_V1);
	KUNIT_EXPECT_EQ(test, sources[1], MT7932_POLICY_WORLD);
	KUNIT_ASSERT_EQ(test, mt7932_policy_candidates("00", sources), 1);
	KUNIT_EXPECT_EQ(test, sources[0], MT7932_POLICY_WORLD);
	mt7932_policy_path(path, sizeof(path), MT7932_POLICY_V1, "AU");
	KUNIT_EXPECT_STREQ(test, path, "mediatek/mt7932/policy/AU.bin");
	mt7932_policy_path(path, sizeof(path), MT7932_POLICY_WORLD, "AU");
	KUNIT_EXPECT_STREQ(test, path, "mediatek/mt7932/policy/world-XZ.bin");

	/* Nothing installed: every candidate is asked for, in order. */
	KUNIT_EXPECT_EQ(test, select_run(env, "NZ"), -ENOENT);
	KUNIT_ASSERT_EQ(test, env->fs.requests, 2);
	KUNIT_EXPECT_STREQ(test, env->fs.log[0], "mediatek/mt7932/policy/NZ.bin");
	KUNIT_EXPECT_STREQ(test, env->fs.log[1], "mediatek/mt7932/policy/world-XZ.bin");
	KUNIT_EXPECT_PTR_EQ(test, env->file.handle, NULL);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_NONE);
	KUNIT_EXPECT_EQ(test, env->choice.n_tried, 2);

	/* cfg80211's numeric codes have no package of their own. */
	KUNIT_EXPECT_EQ(test, mt7932_policy_candidates("98", sources), 1);
	KUNIT_EXPECT_EQ(test, mt7932_policy_candidates("99", sources), 1);
	KUNIT_EXPECT_FALSE(test, mt7932_policy_country("00"));
	KUNIT_EXPECT_FALSE(test, mt7932_policy_country("aU"));
	KUNIT_EXPECT_TRUE(test, mt7932_policy_country("AU"));

	/* 00 never opens a country file. */
	KUNIT_EXPECT_EQ(test, select_run(env, "00"), -ENOENT);
	KUNIT_ASSERT_EQ(test, env->fs.requests, 1);
	KUNIT_EXPECT_STREQ(test, env->fs.log[0], "mediatek/mt7932/policy/world-XZ.bin");
	KUNIT_EXPECT_EQ(test, env->fs.held, 0);
}

/* The country's own package wins; an unusable one gives way to the world. */
static void mt7932_policy_version_test(struct kunit *test)
{
	struct select_env *env = select_env(test);
	size_t v1_size, xz_size, big_size = 16385;
	u8 *v1, *other, *xz, *big;

	v1 = pkg_build(test, "AU", &v1_size);
	xz = pkg_build(test, "XZ", &xz_size);
	other = kunit_kzalloc(test, v1_size, GFP_KERNEL);
	big = kunit_kzalloc(test, big_size, GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, other);
	KUNIT_ASSERT_NOT_NULL(test, big);

	fake_add(&env->fs, "mediatek/mt7932/policy/AU.bin", v1, v1_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);
	KUNIT_ASSERT_EQ(test, select_run(env, "AU"), 0);
	KUNIT_EXPECT_EQ(test, env->fs.requests, 1);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_V1);
	KUNIT_EXPECT_EQ(test, env->choice.version, 1);
	KUNIT_EXPECT_EQ(test, env->choice.size, v1_size);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, 0);
	KUNIT_EXPECT_MEMEQ(test, env->choice.domain, "AU", 2);
	KUNIT_EXPECT_PTR_EQ(test, env->file.data, (const u8 *)v1);
	KUNIT_EXPECT_EQ(test, env->fs.held, 1);

	/* A package of another J7RP version is not used. */
	memcpy(other, v1, v1_size);
	put_unaligned_le16(2, other + 4);
	env->fs.files = 0;
	fake_add(&env->fs, "mediatek/mt7932/policy/AU.bin", other, v1_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);
	KUNIT_ASSERT_EQ(test, select_run(env, "AU"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.tried[0].error, -EINVAL);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, -EINVAL);

	/* Over 16 KiB. */
	memcpy(big, v1, v1_size);
	put_unaligned_le32(big_size, big + 12);
	env->fs.files = 0;
	fake_add(&env->fs, "mediatek/mt7932/policy/AU.bin", big, big_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);
	KUNIT_ASSERT_EQ(test, select_run(env, "AU"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.tried[0].error, -EINVAL);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);

	/* Another country's package under this country's name. */
	env->fs.files = 0;
	fake_add(&env->fs, "mediatek/mt7932/policy/NZ.bin", v1, v1_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);
	KUNIT_ASSERT_EQ(test, select_run(env, "NZ"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, -EINVAL);

	/* A truncated world package leaves nothing to fall back to. */
	env->fs.files = 0;
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size - 1);
	KUNIT_EXPECT_EQ(test, select_run(env, "NZ"), -EINVAL);
	KUNIT_EXPECT_PTR_EQ(test, env->file.handle, NULL);
	KUNIT_EXPECT_EQ(test, env->fs.held, 0);
}

/* The world fallback keeps the country and intersects the channels. */
static void mt7932_policy_fallback_test(struct kunit *test)
{
	struct select_env *env = select_env(test);
	struct mt7932_policy world;
	size_t xz_size;
	unsigned int i;
	u8 *xz;

	xz = pkg_build(test, "XZ", &xz_size);
	KUNIT_ASSERT_EQ(test, mt7932_policy_parse(&world, xz, xz_size, "XZ"), 0);
	/* The world profile forbids 149; the country lists it. */
	memset(pkg_entry(&world, 149) + 1, MT7932_POLICY_NOT_PERMITTED, PKG_ENTRY_SIZE - 1);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);

	KUNIT_ASSERT_EQ(test, select_run(env, "CA"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, -ENOENT);
	KUNIT_EXPECT_EQ(test, env->choice.version, 1);
	KUNIT_EXPECT_MEMEQ(test, env->choice.alpha2, "CA", 2);
	KUNIT_EXPECT_MEMEQ(test, env->reg.alpha2, "CA", 2);
	KUNIT_EXPECT_MEMEQ(test, env->reg.domain, "XZ", 2);
	KUNIT_EXPECT_MEMEQ(test, env->choice.domain, "XZ", 2);
	/* 1, 6, 13 and 36 remain; 149 is gone. */
	KUNIT_EXPECT_EQ(test, env->reg.domain[8], 3);
	KUNIT_EXPECT_EQ(test, env->reg.domain[9], 1);
	KUNIT_ASSERT_EQ(test, env->reg.length, 12 + 4 * 8);
	/* The world profile permits 14 and 165, which the country does not
	 * list: the fallback never adds a channel.
	 */
	KUNIT_EXPECT_TRUE(test, mt7932_policy_permits(&env->policy, 14));
	KUNIT_EXPECT_TRUE(test, mt7932_policy_permits(&env->policy, 165));
	for (i = 0; i < 4; i++) {
		u16 ch = get_unaligned_le16(env->reg.domain + 12 + i * 8);

		KUNIT_EXPECT_TRUE(test, ch != 14 && ch != 149 && ch != 165);
	}

	/* A numeric cfg80211 code falls back with the reason recorded. */
	KUNIT_ASSERT_EQ(test, select_run(env, "98"), 0);
	KUNIT_EXPECT_EQ(test, env->fs.requests, 1);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, -EOPNOTSUPP);
	KUNIT_EXPECT_MEMEQ(test, env->reg.domain, "XZ", 2);

	/* 00 uses the same package as its own: no fallback is reported. */
	KUNIT_ASSERT_EQ(test, select_run(env, "00"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_WORLD);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, 0);
	KUNIT_EXPECT_MEMEQ(test, env->reg.domain, "XZ", 2);
	fake_put(&env->fs, &env->file);
	KUNIT_EXPECT_EQ(test, env->fs.held, 0);
}

/* Country changes in a row, as the notifier drives them without a reload. */
static void mt7932_policy_switch_test(struct kunit *test)
{
	static const struct {
		const char *alpha2;
		enum mt7932_policy_source source;
		u8 domain[2];
		int fallback;
	} steps[] = {
		{ "AU", MT7932_POLICY_V1, "AU", 0 },
		{ "US", MT7932_POLICY_V1, "US", 0 },
		{ "CA", MT7932_POLICY_WORLD, "XZ", -ENOENT },
		{ "00", MT7932_POLICY_WORLD, "XZ", 0 },
		{ "AU", MT7932_POLICY_V1, "AU", 0 },
		{ "US", MT7932_POLICY_V1, "US", 0 },
	};
	struct select_env *env = select_env(test);
	u8 digest[3][SHA256_DIGEST_SIZE];
	size_t au_size, us_size, ca_size, xz_size;
	u8 *au, *us, *ca, *xz;
	unsigned int i;

	au = pkg_build(test, "AU", &au_size);
	us = pkg_build(test, "US", &us_size);
	ca = pkg_build(test, "CA", &ca_size);
	xz = pkg_build(test, "XZ", &xz_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/AU.bin", au, au_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/US.bin", us, us_size);
	fake_add(&env->fs, "mediatek/mt7932/policy/world-XZ.bin", xz, xz_size);
	sha256(au, au_size, digest[0]);
	sha256(us, us_size, digest[1]);
	sha256(xz, xz_size, digest[2]);

	for (i = 0; i < ARRAY_SIZE(steps); i++) {
		const u8 *expected = !memcmp(steps[i].domain, "AU", 2) ? digest[0] :
				     !memcmp(steps[i].domain, "US", 2) ? digest[1] : digest[2];

		KUNIT_ASSERT_EQ_MSG(test, select_run(env, steps[i].alpha2), 0, "step %u", i);
		KUNIT_EXPECT_EQ_MSG(test, env->choice.source, steps[i].source, "step %u", i);
		KUNIT_EXPECT_MEMEQ_MSG(test, env->choice.domain, steps[i].domain, 2,
				       "step %u", i);
		KUNIT_EXPECT_EQ_MSG(test, env->choice.fallback, steps[i].fallback, "step %u", i);
		KUNIT_EXPECT_EQ_MSG(test, env->fs.held, 1, "step %u", i);
		KUNIT_EXPECT_MEMEQ_MSG(test, env->choice.sha256, expected, SHA256_DIGEST_SIZE,
				       "step %u", i);
	}
	/* A package installed later is used on the next change. */
	fake_add(&env->fs, "mediatek/mt7932/policy/CA.bin", ca, ca_size);
	KUNIT_ASSERT_EQ(test, select_run(env, "CA"), 0);
	KUNIT_EXPECT_EQ(test, env->choice.source, MT7932_POLICY_V1);
	KUNIT_EXPECT_EQ(test, env->choice.fallback, 0);
	KUNIT_EXPECT_MEMEQ(test, env->choice.domain, "CA", 2);
	fake_put(&env->fs, &env->file);
	KUNIT_EXPECT_EQ(test, env->fs.held, 0);
}

static struct kunit_case mt7932_policy_test_cases[] = {
	KUNIT_CASE(mt7932_reg_relaxation_test),
	KUNIT_CASE(mt7932_policy_valid_test),
	KUNIT_CASE(mt7932_policy_country_mismatch_test),
	KUNIT_CASE(mt7932_policy_truncation_test),
	KUNIT_CASE(mt7932_policy_header_test),
	KUNIT_CASE(mt7932_policy_table_test),
	KUNIT_CASE(mt7932_policy_world_order_test),
	KUNIT_CASE(mt7932_policy_permits_test),
	KUNIT_CASE(mt7932_policy_filter_test),
	KUNIT_CASE(mt7932_policy_corruption_test),
	KUNIT_CASE(mt7932_policy_order_test),
	KUNIT_CASE(mt7932_policy_version_test),
	KUNIT_CASE(mt7932_policy_fallback_test),
	KUNIT_CASE(mt7932_policy_switch_test),
	{}
};

static struct kunit_suite mt7932_policy_test_suite = {
	.name = "mt7932-regulatory",
	.test_cases = mt7932_policy_test_cases,
};

kunit_test_suite(mt7932_policy_test_suite);
