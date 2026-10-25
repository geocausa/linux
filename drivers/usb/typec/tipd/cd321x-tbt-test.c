// SPDX-License-Identifier: GPL-2.0-only
/*
 * KUnit tests for the CD321x USB4/Thunderbolt router lookup
 */

#include <kunit/device.h>
#include <kunit/static_stub.h>
#include <kunit/test.h>
#include <kunit/test-bug.h>
#include <linux/module.h>
#include <linux/property.h>
#include <linux/usb/typec_mux.h>

#include "tps6598x.h"

/*
 * A Type-C connector whose port@2 is linked to a USB4 router, as on the M1
 * and M2 machines, and a connector without such a link.
 */
static const struct software_node cd321x_test_router_ep;
static const struct software_node cd321x_test_connector_ep;

static const struct property_entry cd321x_test_router_props[] = {
	PROPERTY_ENTRY_BOOL("thunderbolt-switch"),
	{ }
};

static const struct software_node cd321x_test_router = {
	.name = "cd321x-test-router",
	.properties = cd321x_test_router_props,
};

static const struct software_node cd321x_test_router_port = {
	.name = "port@1",
	.parent = &cd321x_test_router,
};

static const struct software_node_ref_args cd321x_test_router_refs[] = {
	SOFTWARE_NODE_REFERENCE(&cd321x_test_connector_ep),
};

static const struct property_entry cd321x_test_router_ep_props[] = {
	PROPERTY_ENTRY_REF_ARRAY("remote-endpoint", cd321x_test_router_refs),
	{ }
};

static const struct software_node cd321x_test_router_ep = {
	.name = "endpoint",
	.parent = &cd321x_test_router_port,
	.properties = cd321x_test_router_ep_props,
};

static const struct software_node cd321x_test_connector = {
	.name = "cd321x-test-connector",
};

static const struct software_node cd321x_test_connector_port = {
	.name = "port@2",
	.parent = &cd321x_test_connector,
};

static const struct software_node_ref_args cd321x_test_connector_refs[] = {
	SOFTWARE_NODE_REFERENCE(&cd321x_test_router_ep),
};

static const struct property_entry cd321x_test_connector_ep_props[] = {
	PROPERTY_ENTRY_REF_ARRAY("remote-endpoint", cd321x_test_connector_refs),
	{ }
};

static const struct software_node cd321x_test_connector_ep = {
	.name = "endpoint",
	.parent = &cd321x_test_connector_port,
	.properties = cd321x_test_connector_ep_props,
};

static const struct software_node cd321x_test_plain_connector = {
	.name = "cd321x-test-plain-connector",
};

static const struct software_node * const cd321x_test_nodes[] = {
	&cd321x_test_router,
	&cd321x_test_router_port,
	&cd321x_test_router_ep,
	&cd321x_test_connector,
	&cd321x_test_connector_port,
	&cd321x_test_connector_ep,
	&cd321x_test_plain_connector,
	NULL
};

struct cd321x_tbt_test {
	struct device *dev;
	bool keep_waiting;
	unsigned int keep_waiting_calls;
};

/* Stands in for the driver core's answer on whether to keep waiting. */
static bool cd321x_tbt_test_keep_waiting(struct device *dev)
{
	struct cd321x_tbt_test *ctx = kunit_get_current_test()->priv;

	ctx->keep_waiting_calls++;
	return ctx->keep_waiting;
}

static void cd321x_tbt_test_unregister_nodes(void *unused)
{
	software_node_unregister_node_group(cd321x_test_nodes);
}

static int cd321x_tbt_test_init(struct kunit *test)
{
	struct cd321x_tbt_test *ctx;
	int ret;

	ctx = kunit_kzalloc(test, sizeof(*ctx), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, ctx);
	test->priv = ctx;

	ret = software_node_register_node_group(cd321x_test_nodes);
	KUNIT_ASSERT_EQ(test, ret, 0);
	ret = kunit_add_action_or_reset(test, cd321x_tbt_test_unregister_nodes,
					NULL);
	KUNIT_ASSERT_EQ(test, ret, 0);

	ctx->dev = kunit_device_register(test, "cd321x");
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, ctx->dev);

	kunit_activate_static_stub(test, cd321x_tbt_switch_keep_waiting,
				   cd321x_tbt_test_keep_waiting);
	return 0;
}

static int cd321x_tbt_test_set(struct typec_thunderbolt_switch_dev *sw,
			       const struct typec_thunderbolt_switch_data *data)
{
	return 0;
}

static void cd321x_tbt_test_unregister_switch(void *sw)
{
	typec_thunderbolt_switch_unregister(sw);
}

/* A connector without a router link never waits. */
static void cd321x_tbt_no_router_test(struct kunit *test)
{
	struct cd321x_tbt_test *ctx = test->priv;
	struct typec_thunderbolt_switch *sw;

	sw = cd321x_tbt_switch_get(ctx->dev,
				   software_node_fwnode(&cd321x_test_plain_connector));
	KUNIT_EXPECT_NULL(test, sw);
	KUNIT_EXPECT_EQ(test, ctx->keep_waiting_calls, 0);
}

/* A router whose driver has registered its switch is used. */
static void cd321x_tbt_router_bound_test(struct kunit *test)
{
	struct cd321x_tbt_test *ctx = test->priv;
	struct typec_thunderbolt_switch_desc desc = {
		.fwnode = software_node_fwnode(&cd321x_test_router),
		.set = cd321x_tbt_test_set,
	};
	struct typec_thunderbolt_switch_dev *sw_dev;
	struct typec_thunderbolt_switch *sw;
	struct device *router;
	int ret;

	router = kunit_device_register(test, "router");
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, router);
	/* The lookup takes a reference on the router driver's module. */
	KUNIT_ASSERT_NOT_NULL(test, router->driver);

	sw_dev = typec_thunderbolt_switch_register(router, &desc);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, sw_dev);
	ret = kunit_add_action_or_reset(test, cd321x_tbt_test_unregister_switch,
					sw_dev);
	KUNIT_ASSERT_EQ(test, ret, 0);

	sw = cd321x_tbt_switch_get(ctx->dev,
				   software_node_fwnode(&cd321x_test_connector));
	KUNIT_EXPECT_NOT_ERR_OR_NULL(test, sw);
	KUNIT_EXPECT_EQ(test, ctx->keep_waiting_calls, 0);
	typec_thunderbolt_switch_put(sw);
}

/* A router that has not registered its switch yet defers the port. */
static void cd321x_tbt_router_pending_test(struct kunit *test)
{
	struct cd321x_tbt_test *ctx = test->priv;
	struct typec_thunderbolt_switch *sw;

	ctx->keep_waiting = true;
	sw = cd321x_tbt_switch_get(ctx->dev,
				   software_node_fwnode(&cd321x_test_connector));
	KUNIT_EXPECT_PTR_EQ(test, sw, ERR_PTR(-EPROBE_DEFER));
	KUNIT_EXPECT_EQ(test, ctx->keep_waiting_calls, 1);
}

/*
 * Once the driver core stops waiting for missing dependencies, a router that
 * never registered its switch, for example because its driver is not built,
 * no longer keeps the port from probing.
 */
static void cd321x_tbt_router_never_binds_test(struct kunit *test)
{
	struct cd321x_tbt_test *ctx = test->priv;
	struct typec_thunderbolt_switch *sw;

	ctx->keep_waiting = false;
	sw = cd321x_tbt_switch_get(ctx->dev,
				   software_node_fwnode(&cd321x_test_connector));
	KUNIT_EXPECT_NULL(test, sw);
	KUNIT_EXPECT_EQ(test, ctx->keep_waiting_calls, 1);
}

static struct kunit_case cd321x_tbt_cases[] = {
	KUNIT_CASE(cd321x_tbt_no_router_test),
	KUNIT_CASE(cd321x_tbt_router_bound_test),
	KUNIT_CASE(cd321x_tbt_router_pending_test),
	KUNIT_CASE(cd321x_tbt_router_never_binds_test),
	{}
};

static struct kunit_suite cd321x_tbt_suite = {
	.name = "cd321x-tbt",
	.init = cd321x_tbt_test_init,
	.test_cases = cd321x_tbt_cases,
};

kunit_test_suite(cd321x_tbt_suite);
MODULE_DESCRIPTION("KUnit tests for the CD321x USB4/Thunderbolt router lookup");
MODULE_LICENSE("GPL");
MODULE_IMPORT_NS("EXPORTED_FOR_KUNIT_TESTING");
