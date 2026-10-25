// SPDX-License-Identifier: GPL-2.0
/*
 * KUnit tests for usb_hub_port_power_off() and usb_hub_port_power_on().
 *
 * The tests run on a host controller made of software: a USB2 and a
 * SuperSpeed root hub with one port each, peered as on an xHCI, so that the
 * hub driver, the port devices and their runtime PM are the real ones. The
 * controller only keeps the status of its ports, and no device ever connects.
 * Port attributes are written through sysfs, as userspace writes them.
 */

#include <kunit/device.h>
#include <kunit/test.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mount.h>
#include <linux/pm_runtime.h>
#include <linux/slab.h>
#include <linux/unaligned.h>
#include <linux/usb.h>
#include <linux/usb/ch11.h>
#include <linux/usb/hcd.h>

struct port_power_test_hc {
	spinlock_t lock;	/* protects the port status, as hub_wq reads it too */
	u16 status;
	u16 change;
	unsigned int power_ons;
	bool fail_power_off;
};

struct port_power_test {
	struct usb_hcd *hcd;
	struct usb_hcd *shared_hcd;
	struct usb_device *hs_root;
	struct usb_device *ss_root;
	struct device *hs_port;
	struct device *ss_port;
	struct vfsmount *sysfs;
};

static const struct {
	struct usb_bos_descriptor bos;
	struct usb_ss_cap_descriptor ss_cap;
} __packed port_power_test_bos = {
	.bos = {
		.bLength = USB_DT_BOS_SIZE,
		.bDescriptorType = USB_DT_BOS,
		.wTotalLength = cpu_to_le16(sizeof(port_power_test_bos)),
		.bNumDeviceCaps = 1,
	},
	.ss_cap = {
		.bLength = USB_DT_USB_SS_CAP_SIZE,
		.bDescriptorType = USB_DT_DEVICE_CAPABILITY,
		.bDevCapabilityType = USB_SS_CAP_TYPE,
		.wSpeedSupported = cpu_to_le16(USB_5GBPS_OPERATION),
		.bFunctionalitySupport = USB_LOW_SPEED_OPERATION,
	},
};

static struct port_power_test_hc *port_power_test_hc(struct usb_hcd *hcd)
{
	return (struct port_power_test_hc *)hcd->hcd_priv;
}

static u16 port_power_test_power_bit(struct usb_hcd *hcd)
{
	return hcd->speed >= HCD_USB3 ? USB_SS_PORT_STAT_POWER : USB_PORT_STAT_POWER;
}

static void port_power_test_set_power(struct usb_hcd *hcd, bool on)
{
	struct port_power_test_hc *hc = port_power_test_hc(hcd);

	if (on) {
		hc->status |= port_power_test_power_bit(hcd);
		hc->power_ons++;
	} else {
		hc->status &= ~port_power_test_power_bit(hcd);
	}
	if (hcd->speed >= HCD_USB3) {
		hc->status &= ~USB_PORT_STAT_LINK_STATE;
		hc->status |= on ? USB_SS_PORT_LS_RX_DETECT : USB_SS_PORT_LS_SS_DISABLED;
	}
}

static bool port_power_test_powered(struct usb_hcd *hcd)
{
	struct port_power_test_hc *hc = port_power_test_hc(hcd);

	guard(spinlock_irqsave)(&hc->lock);
	return hc->status & port_power_test_power_bit(hcd);
}

static unsigned int port_power_test_power_ons(struct usb_hcd *hcd)
{
	struct port_power_test_hc *hc = port_power_test_hc(hcd);

	guard(spinlock_irqsave)(&hc->lock);
	return hc->power_ons;
}

static void port_power_test_hub_descriptor(struct usb_hcd *hcd, struct usb_hub_descriptor *desc)
{
	memset(desc, 0, sizeof(*desc));
	desc->bNbrPorts = 1;
	desc->wHubCharacteristics = cpu_to_le16(HUB_CHAR_INDV_PORT_LPSM | HUB_CHAR_NO_OCPM);
	desc->bPwrOn2PwrGood = 1;
	if (hcd->speed >= HCD_USB3) {
		desc->bDescriptorType = USB_DT_SS_HUB;
		desc->bDescLength = USB_DT_SS_HUB_SIZE;
	} else {
		desc->bDescriptorType = USB_DT_HUB;
		desc->bDescLength = USB_DT_HUB_NONVAR_SIZE + 2;
		desc->u.hs.DeviceRemovable[1] = 0xff;
	}
}

static u16 port_power_test_change_bit(u16 feature)
{
	switch (feature) {
	case USB_PORT_FEAT_C_CONNECTION:
		return USB_PORT_STAT_C_CONNECTION;
	case USB_PORT_FEAT_C_ENABLE:
		return USB_PORT_STAT_C_ENABLE;
	case USB_PORT_FEAT_C_RESET:
		return USB_PORT_STAT_C_RESET;
	case USB_PORT_FEAT_C_BH_PORT_RESET:
		return USB_PORT_STAT_C_BH_RESET;
	case USB_PORT_FEAT_C_PORT_LINK_STATE:
		return USB_PORT_STAT_C_LINK_STATE;
	}
	return 0;
}

static int port_power_test_hub_control(struct usb_hcd *hcd, u16 type_req, u16 value, u16 index,
				       char *buf, u16 length)
{
	struct port_power_test_hc *hc = port_power_test_hc(hcd);

	switch (type_req) {
	case DeviceRequest | USB_REQ_GET_DESCRIPTOR:
		if (hcd->speed < HCD_USB3 || value >> 8 != USB_DT_BOS)
			return -EPIPE;
		length = min_t(u16, length, sizeof(port_power_test_bos));
		memcpy(buf, &port_power_test_bos, length);
		return length;
	case GetHubDescriptor:
		port_power_test_hub_descriptor(hcd, (struct usb_hub_descriptor *)buf);
		return 0;
	case GetHubStatus:
		memset(buf, 0, 4);
		return 0;
	}

	if ((index & 0xff) != 1)
		return -EPIPE;

	guard(spinlock_irqsave)(&hc->lock);
	switch (type_req) {
	case GetPortStatus:
		put_unaligned_le16(hc->status, buf);
		put_unaligned_le16(hc->change, buf + 2);
		return 0;
	case SetPortFeature:
		if (value == USB_PORT_FEAT_POWER)
			port_power_test_set_power(hcd, true);
		else if (value == USB_PORT_FEAT_RESET)
			hc->change |= USB_PORT_STAT_C_RESET;
		else if (value == USB_PORT_FEAT_BH_PORT_RESET)
			hc->change |= USB_PORT_STAT_C_RESET | USB_PORT_STAT_C_BH_RESET;
		return 0;
	case ClearPortFeature:
		if (value == USB_PORT_FEAT_POWER && hc->fail_power_off) {
			hc->fail_power_off = false;
			return -ENOMEM;
		}
		if (value == USB_PORT_FEAT_POWER)
			port_power_test_set_power(hcd, false);
		else
			hc->change &= ~port_power_test_change_bit(value);
		return 0;
	}
	return -EPIPE;
}

static int port_power_test_reset(struct usb_hcd *hcd)
{
	struct port_power_test_hc *hc = port_power_test_hc(hcd);

	/* As on an xHCI, the primary HCD has the USB2 root hub */
	if (usb_hcd_is_primary_hcd(hcd)) {
		hcd->speed = HCD_USB2;
		hcd->self.root_hub->speed = USB_SPEED_HIGH;
	}
	spin_lock_init(&hc->lock);
	port_power_test_set_power(hcd, false);
	hc->power_ons = 0;

	return 0;
}

static int port_power_test_start(struct usb_hcd *hcd)
{
	hcd->uses_new_polling = 1;
	return 0;
}

static void port_power_test_stop(struct usb_hcd *hcd)
{
}

static int port_power_test_urb_enqueue(struct usb_hcd *hcd, struct urb *urb, gfp_t mem_flags)
{
	return -ENODEV;
}

static int port_power_test_urb_dequeue(struct usb_hcd *hcd, struct urb *urb, int status)
{
	return -EINVAL;
}

static int port_power_test_get_frame_number(struct usb_hcd *hcd)
{
	return 0;
}

static int port_power_test_hub_status_data(struct usb_hcd *hcd, char *buf)
{
	return 0;
}

static const struct hc_driver port_power_test_hc_driver = {
	.description = "usb-port-power-test",
	.product_desc = "USB port power test controller",
	.hcd_priv_size = sizeof(struct port_power_test_hc),
	.flags = HCD_USB3 | HCD_SHARED,
	.reset = port_power_test_reset,
	.start = port_power_test_start,
	.stop = port_power_test_stop,
	.urb_enqueue = port_power_test_urb_enqueue,
	.urb_dequeue = port_power_test_urb_dequeue,
	.get_frame_number = port_power_test_get_frame_number,
	.hub_status_data = port_power_test_hub_status_data,
	.hub_control = port_power_test_hub_control,
};

static void port_power_test_put_hcds(void *data)
{
	struct port_power_test *t = data;

	usb_put_hcd(t->shared_hcd);
	usb_put_hcd(t->hcd);
}

static void port_power_test_remove_hcds(void *data)
{
	struct port_power_test *t = data;

	put_device(t->ss_port);
	put_device(t->hs_port);
	usb_remove_hcd(t->shared_hcd);
	usb_remove_hcd(t->hcd);
}

static void port_power_test_unmount(void *data)
{
	kern_unmount(data);
}

/* The hub driver has set up a root hub once it waits for status changes. */
static void port_power_test_wait_hub(struct kunit *test, struct usb_hcd *hcd)
{
	unsigned int ms;

	for (ms = 0; ms < 2000 && !READ_ONCE(hcd->status_urb); ms += 20)
		msleep(20);
	KUNIT_ASSERT_NOT_NULL(test, READ_ONCE(hcd->status_urb));
}

static struct device *port_power_test_port(struct kunit *test, struct usb_device *root)
{
	char *name = kasprintf(GFP_KERNEL, "%s-port1", dev_name(&root->dev));
	struct device *port;

	KUNIT_ASSERT_NOT_NULL(test, name);
	port = device_find_child_by_name(&root->actconfig->interface[0]->dev, name);
	kfree(name);
	KUNIT_ASSERT_NOT_NULL(test, port);

	return port;
}

static int port_power_test_init(struct kunit *test)
{
	struct file_system_type *sysfs_type;
	struct port_power_test *t;
	struct device *dev;
	int ret;

	t = kunit_kzalloc(test, sizeof(*t), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, t);
	test->priv = t;

	dev = kunit_device_register(test, "usb-port-power-test");
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, dev);

	t->hcd = usb_create_hcd(&port_power_test_hc_driver, dev, dev_name(dev));
	KUNIT_ASSERT_NOT_NULL(test, t->hcd);
	t->hcd->skip_phy_initialization = 1;
	t->shared_hcd = usb_create_shared_hcd(&port_power_test_hc_driver, dev, dev_name(dev),
					      t->hcd);
	if (!t->shared_hcd)
		usb_put_hcd(t->hcd);
	KUNIT_ASSERT_NOT_NULL(test, t->shared_hcd);
	ret = kunit_add_action_or_reset(test, port_power_test_put_hcds, t);
	KUNIT_ASSERT_EQ(test, ret, 0);

	ret = usb_add_hcd(t->hcd, 0, 0);
	KUNIT_ASSERT_EQ(test, ret, 0);
	ret = usb_add_hcd(t->shared_hcd, 0, 0);
	if (ret)
		usb_remove_hcd(t->hcd);
	KUNIT_ASSERT_EQ(test, ret, 0);
	ret = kunit_add_action_or_reset(test, port_power_test_remove_hcds, t);
	KUNIT_ASSERT_EQ(test, ret, 0);

	t->hs_root = t->hcd->self.root_hub;
	t->ss_root = t->shared_hcd->self.root_hub;
	port_power_test_wait_hub(test, t->hcd);
	port_power_test_wait_hub(test, t->shared_hcd);
	t->hs_port = port_power_test_port(test, t->hs_root);
	t->ss_port = port_power_test_port(test, t->ss_root);
	KUNIT_ASSERT_TRUE(test, port_power_test_powered(t->hcd));
	KUNIT_ASSERT_TRUE(test, port_power_test_powered(t->shared_hcd));

	/* Root hub runtime PM is not what these tests are about. */
	usb_disable_autosuspend(t->hs_root);
	usb_disable_autosuspend(t->ss_root);

	/* sysfs is built in, so its type holds no module reference. */
	sysfs_type = get_fs_type("sysfs");
	KUNIT_ASSERT_NOT_NULL(test, sysfs_type);
	t->sysfs = kern_mount(sysfs_type);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, t->sysfs);
	ret = kunit_add_action_or_reset(test, port_power_test_unmount, t->sysfs);
	KUNIT_ASSERT_EQ(test, ret, 0);

	return 0;
}

/* Write @val to the attribute @attr of @dev, as userspace does. */
static int port_power_test_write(struct kunit *test, struct device *dev, const char *attr,
				 const char *val)
{
	struct port_power_test *t = test->priv;
	struct path root = { .mnt = t->sysfs, .dentry = t->sysfs->mnt_root };
	char *kpath = kobject_get_path(&dev->kobj, GFP_KERNEL);
	struct file *file;
	loff_t pos = 0;
	ssize_t ret;
	char *path;

	KUNIT_ASSERT_NOT_NULL(test, kpath);
	path = kasprintf(GFP_KERNEL, "%s/%s", kpath + 1, attr);
	kfree(kpath);
	KUNIT_ASSERT_NOT_NULL(test, path);

	file = file_open_root(&root, path, O_WRONLY, 0);
	kfree(path);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, file);
	ret = kernel_write(file, val, strlen(val), &pos);
	fput(file);

	return ret < 0 ? ret : 0;
}

/* Let userspace allow runtime PM to switch off both ports. */
static void port_power_test_allow_power_off(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	if (!IS_ENABLED(CONFIG_PM))
		kunit_skip(test, "needs CONFIG_PM");

	KUNIT_ASSERT_EQ(test, port_power_test_write(test, t->hs_port, "power/pm_qos_no_power_off",
						    "0"), 0);
	KUNIT_ASSERT_EQ(test, port_power_test_write(test, t->ss_port, "power/pm_qos_no_power_off",
						    "0"), 0);
}

static int port_power_test_usage(struct device *dev)
{
	return atomic_read(&dev->power.usage_count);
}

/* The port stays off until powered on, and only the given port is switched. */
static void port_power_test_hold(struct kunit *test)
{
	struct port_power_test *t = test->priv;
	unsigned int hs_power_ons = port_power_test_power_ons(t->hcd);
	int usage = port_power_test_usage(t->ss_port);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	/* A second change of the lanes finds the port held for it */
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage);

	/* Nothing is held any more */
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 1);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 2), -EINVAL);
	usb_unlock_device(t->ss_root);

	/* The USB2 root port was never switched */
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->hcd));
	KUNIT_EXPECT_EQ(test, port_power_test_power_ons(t->hcd), hs_power_ons);
}

/*
 * With port power-off allowed, port runtime PM runs while the caller holds
 * the hub: it can neither suspend the held port nor resume it, which would
 * switch its power on, and neither can the resume of its USB2 peer.
 */
static void port_power_test_runtime_pm(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	port_power_test_allow_power_off(test);
	/* The USB2 port, while active, keeps its SuperSpeed peer active */
	KUNIT_ASSERT_EQ(test, pm_runtime_resume_and_get(t->hs_port), 0);
	KUNIT_ASSERT_TRUE(test, pm_runtime_active(t->ss_port));

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);

	/* The USB2 port suspends, which lets its peer suspend */
	pm_runtime_put_sync(t->hs_port);
	KUNIT_EXPECT_TRUE(test, pm_runtime_suspended(t->hs_port));
	KUNIT_EXPECT_LT(test, pm_runtime_suspend(t->ss_port), 0);
	KUNIT_EXPECT_TRUE(test, pm_runtime_active(t->ss_port));

	/* The USB2 port resumes, and resumes its peer first */
	KUNIT_ASSERT_EQ(test, pm_runtime_resume_and_get(t->hs_port), 0);
	KUNIT_EXPECT_GE(test, pm_runtime_resume(t->ss_port), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_TRUE(test, pm_runtime_active(t->ss_port));
	usb_unlock_device(t->ss_root);
	pm_runtime_put(t->hs_port);
}

/* A port that runtime PM switched off is the caller's to hold, not someone else's */
static void port_power_test_runtime_suspended(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	port_power_test_allow_power_off(test);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->hs_port), 0);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->ss_port), 0);
	KUNIT_ASSERT_TRUE(test, pm_runtime_suspended(t->ss_port));
	KUNIT_ASSERT_FALSE(test, port_power_test_powered(t->shared_hcd));

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_TRUE(test, pm_runtime_active(t->ss_port));

	/* The resume of the USB2 peer leaves it off */
	KUNIT_ASSERT_EQ(test, pm_runtime_resume_and_get(t->hs_port), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	/* and only the caller switches it on */
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);
	pm_runtime_put(t->hs_port);
}

/* The "disable" attribute can keep a held port off, and then owns it */
static void port_power_test_disable(struct kunit *test)
{
	struct port_power_test *t = test->priv;
	int usage = port_power_test_usage(t->ss_port);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	usb_unlock_device(t->ss_root);

	/* Not powered on behind the lanes' back */
	KUNIT_EXPECT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "0"), -EBUSY);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	/* Kept off: from now on it is userspace's port */
	KUNIT_EXPECT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "1"), 0);
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 1);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);

	KUNIT_EXPECT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "0"), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
}

/* A disabled port stays userspace's after runtime PM has suspended it */
static void port_power_test_disabled_suspended(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	KUNIT_ASSERT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "1"), 0);
	port_power_test_allow_power_off(test);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->hs_port), 0);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->ss_port), 0);
	KUNIT_ASSERT_TRUE(test, pm_runtime_suspended(t->ss_port));

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 1);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);
}

/* The host controller can go away while a port is held off */
static void port_power_test_removed_held(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	usb_unlock_device(t->ss_root);
}

/* Review: admin-disabled SS port must stay off while the mux changes lanes. */
static void port_power_review_disabled_peer_resume(struct kunit *test)
{
	struct port_power_test *t = test->priv;

	KUNIT_ASSERT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "1"), 0);
	port_power_test_allow_power_off(test);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->hs_port), 0);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->ss_port), 0);
	KUNIT_ASSERT_TRUE(test, pm_runtime_suspended(t->ss_port));
	KUNIT_ASSERT_FALSE(test, port_power_test_powered(t->shared_hcd));

	/* The same root-hub lock the mux holds over the PIPE/lane change. */
	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	/* A peer resume is not excluded by that root-hub lock. */
	KUNIT_ASSERT_GE(test, pm_runtime_resume_and_get(t->hs_port), 0);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);
	pm_runtime_put(t->hs_port);
}

/* Review: a failed takeover request must not consume the driver's PM hold. */
static void port_power_review_failed_takeover(struct kunit *test)
{
	struct port_power_test *t = test->priv;
	struct port_power_test_hc *hc = port_power_test_hc(t->shared_hcd);
	int usage = port_power_test_usage(t->ss_port);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	usb_unlock_device(t->ss_root);
	scoped_guard(spinlock_irqsave, &hc->lock)
		hc->fail_power_off = true;
	KUNIT_EXPECT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "1"), -ENOMEM);
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage + 1);
	usb_lock_device(t->ss_root);
	/* The next PHY update must still own its restoration. */
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);
}

/* A previously repowered administrative port must be off before moving lanes. */
static void port_power_test_disabled_repowered(struct kunit *test)
{
	struct port_power_test *t = test->priv;
	struct port_power_test_hc *hc = port_power_test_hc(t->shared_hcd);
	int usage = port_power_test_usage(t->ss_port);

	KUNIT_ASSERT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "1"), 0);
	/* Model power restored behind usbcore's back, with disable still set. */
	scoped_guard(spinlock_irqsave, &hc->lock)
		port_power_test_set_power(t->shared_hcd, true);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));

	/* A failed reassertion must not permit the lane change. */
	scoped_guard(spinlock_irqsave, &hc->lock) {
		port_power_test_set_power(t->shared_hcd, true);
		hc->fail_power_off = true;
	}
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), -ENOMEM);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	usb_unlock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage);

	/* Ending the handover leaves ownership with sysfs. */
	KUNIT_EXPECT_EQ(test, port_power_test_write(test, t->ss_port, "disable", "0"), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
}

/* An unpowered port without an administrative owner still needs a PM hold. */
static void port_power_test_unowned_off(struct kunit *test)
{
	struct port_power_test *t = test->priv;
	int usage;

	port_power_test_allow_power_off(test);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->hs_port), 0);
	KUNIT_ASSERT_GE(test, pm_runtime_suspend(t->ss_port), 0);
	KUNIT_ASSERT_FALSE(test, port_power_test_powered(t->shared_hcd));
	/* Model an active port whose PM is unavailable and PP is still off. */
	pm_runtime_disable(t->ss_port);
	KUNIT_EXPECT_EQ(test, pm_runtime_set_active(t->ss_port), 0);
	usage = port_power_test_usage(t->ss_port);

	usb_lock_device(t->ss_root);
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_off(t->ss_root, 1), 0);
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage + 1);
	KUNIT_EXPECT_FALSE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, usb_hub_port_power_on(t->ss_root, 1), 0);
	KUNIT_EXPECT_TRUE(test, port_power_test_powered(t->shared_hcd));
	KUNIT_EXPECT_EQ(test, port_power_test_usage(t->ss_port), usage);
	usb_unlock_device(t->ss_root);
	pm_runtime_enable(t->ss_port);
}

static struct kunit_case port_power_test_cases[] = {
	KUNIT_CASE(port_power_test_hold),
	KUNIT_CASE(port_power_test_runtime_pm),
	KUNIT_CASE(port_power_test_runtime_suspended),
	KUNIT_CASE(port_power_test_disable),
	KUNIT_CASE(port_power_test_disabled_suspended),
	KUNIT_CASE(port_power_test_removed_held),
	KUNIT_CASE(port_power_review_disabled_peer_resume),
	KUNIT_CASE(port_power_test_disabled_repowered),
	KUNIT_CASE(port_power_test_unowned_off),
	KUNIT_CASE(port_power_review_failed_takeover),
	{}
};

static struct kunit_suite port_power_test_suite = {
	.name = "usb-hub-port-power",
	.init = port_power_test_init,
	.test_cases = port_power_test_cases,
};

kunit_test_suite(port_power_test_suite);

MODULE_DESCRIPTION("KUnit tests for USB hub port power switching");
MODULE_LICENSE("GPL");
