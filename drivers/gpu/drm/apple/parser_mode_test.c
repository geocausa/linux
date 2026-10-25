// SPDX-License-Identifier: GPL-2.0-only OR MIT
/*
 * KUnit tests for the ranking of the timing modes that DCP reports.
 */

#include <kunit/test.h>
#include <linux/bits.h>

#include "parser.h"

struct dcp_mode_rank_case {
	const char *desc;
	bool preferred;
	s64 score;
	bool best_preferred;
	s64 best_score;
	bool above;
};

static const struct dcp_mode_rank_case dcp_mode_rank_cases[] = {
	{ "first mode", false, 0, false, -1, true },
	{ "preferred over a score above bit 32", true, 0, false, BIT_ULL(32) + 1, true },
	{ "score above bit 32 under a preferred mode", false, BIT_ULL(32) + 1, true, 0 },
	{ "preferred with bit 32 already set", true, BIT_ULL(32), false, BIT_ULL(32) + 1, true },
	{ "higher of two preferred scores", true, 5, true, 4, true },
	{ "lower of two preferred scores", true, 4, true, 5 },
	{ "higher of two other scores", false, 7, false, 3, true },
	{ "equal ranks keep the first mode", true, 3, true, 3 },
};

KUNIT_ARRAY_PARAM_DESC(dcp_mode_rank, dcp_mode_rank_cases, desc);

static void dcp_mode_rank_test(struct kunit *test)
{
	const struct dcp_mode_rank_case *c = test->param_value;

	KUNIT_EXPECT_EQ(test, dcp_mode_ranks_above(c->preferred, c->score,
						   c->best_preferred,
						   c->best_score),
			c->above);
}

static struct kunit_case dcp_mode_rank_test_cases[] = {
	KUNIT_CASE_PARAM(dcp_mode_rank_test, dcp_mode_rank_gen_params),
	{}
};

static struct kunit_suite dcp_mode_rank_test_suite = {
	.name = "apple-dcp-mode-rank",
	.test_cases = dcp_mode_rank_test_cases,
};

kunit_test_suite(dcp_mode_rank_test_suite);
