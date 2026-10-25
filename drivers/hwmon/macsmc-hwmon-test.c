// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * KUnit tests for the Apple SMC hwmon driver: channel numbering of the
 * temperature sensors, which thermal zones depend on.
 *
 * Included by macsmc-hwmon.c.
 */

#include <kunit/test.h>

/* What macsmc_hwmon_create_sensor() leaves behind for one node. */
static void macsmc_hwmon_test_add(struct macsmc_hwmon_sensors *temp,
				  const char *key, int create_ret)
{
	struct macsmc_hwmon_sensor *sensor = &temp->sensors[temp->count];

	sensor->macsmc_key = _SMC_KEY(key);
	strscpy_pad(sensor->label, key, sizeof(sensor->label));
	macsmc_hwmon_add_temp_channel(temp, create_ret);
}

/* A node that fails to set up keeps its channel; later channels do not move. */
static void macsmc_hwmon_test_failed_temp_keeps_slot(struct kunit *test)
{
	struct macsmc_hwmon_sensor sensors[3] = {};
	struct macsmc_hwmon hwmon = {
		.temp = { .sensors = sensors },
	};
	u32 configs[ARRAY_SIZE(sensors) + 1] = {};

	macsmc_hwmon_test_add(&hwmon.temp, "Te05", 0);
	macsmc_hwmon_test_add(&hwmon.temp, "Te0S", -EINVAL);
	macsmc_hwmon_test_add(&hwmon.temp, "Tp01", 0);

	KUNIT_ASSERT_EQ(test, hwmon.temp.count, 3);
	KUNIT_EXPECT_EQ(test, sensors[0].macsmc_key, _SMC_KEY("Te05"));
	KUNIT_EXPECT_EQ(test, sensors[2].macsmc_key, _SMC_KEY("Tp01"));

	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_input, 0), 0444);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_input, 1), 0);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_label, 1), 0);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_input, 2), 0444);

	/* A zero entry would end the channel list at the failed sensor. */
	macsmc_hwmon_populate_configs(configs, &hwmon.temp);
	KUNIT_EXPECT_NE(test, configs[0], 0);
	KUNIT_EXPECT_NE(test, configs[1], 0);
	KUNIT_EXPECT_NE(test, configs[2], 0);
	KUNIT_EXPECT_EQ(test, configs[3], 0);
}

static void macsmc_hwmon_test_temp_label(struct kunit *test)
{
	struct macsmc_hwmon_sensor sensors[2] = {};
	struct macsmc_hwmon hwmon = {
		.temp = { .sensors = sensors },
	};

	macsmc_hwmon_test_add(&hwmon.temp, "Tp05", 0);
	sensors[1].macsmc_key = _SMC_KEY("Tp09");
	macsmc_hwmon_add_temp_channel(&hwmon.temp, 0);

	KUNIT_ASSERT_EQ(test, hwmon.temp.count, 2);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_label, 0), 0444);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_label, 1), 0);
	KUNIT_EXPECT_EQ(test, macsmc_hwmon_is_visible(&hwmon, hwmon_temp,
						      hwmon_temp_input, 1), 0444);
}

static struct kunit_case macsmc_hwmon_test_cases[] = {
	KUNIT_CASE(macsmc_hwmon_test_failed_temp_keeps_slot),
	KUNIT_CASE(macsmc_hwmon_test_temp_label),
	{}
};

static struct kunit_suite macsmc_hwmon_test_suite = {
	.name = "macsmc-hwmon",
	.test_cases = macsmc_hwmon_test_cases,
};

kunit_test_suite(macsmc_hwmon_test_suite);
