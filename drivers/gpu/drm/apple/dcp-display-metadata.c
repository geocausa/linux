// SPDX-License-Identifier: GPL-2.0-only OR MIT
#include <linux/err.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <drm/drm_edid.h>
#include <drm/drm_modes.h>

#include "dcp-display-metadata.h"

static int dcp_metadata_timing(struct detailed_timing *timing,
			       const struct drm_display_mode *mode,
			       const struct dcp_display_metadata *metadata)
{
	struct detailed_pixel_timing *pixel = &timing->data.pixel_data;
	u32 hblank, vblank, hoff, voff, hsync, vsync;
	u32 width = metadata->width_cm * 10, height = metadata->height_cm * 10;
	u32 hflags = mode->flags & (DRM_MODE_FLAG_PHSYNC | DRM_MODE_FLAG_NHSYNC);
	u32 vflags = mode->flags & (DRM_MODE_FLAG_PVSYNC | DRM_MODE_FLAG_NVSYNC);

	if (mode->clock <= 0 || mode->clock > 655350 || mode->clock % 10 ||
	    mode->hdisplay <= 0 || mode->vdisplay <= 0 ||
	    mode->hdisplay > 4095 || mode->vdisplay > 4095 ||
	    mode->htotal <= mode->hdisplay || mode->vtotal <= mode->vdisplay ||
	    mode->hsync_start < mode->hdisplay || mode->hsync_end <= mode->hsync_start ||
	    mode->hsync_end > mode->htotal || mode->vsync_start < mode->vdisplay ||
	    mode->vsync_end <= mode->vsync_start || mode->vsync_end > mode->vtotal ||
	    (hflags != DRM_MODE_FLAG_PHSYNC && hflags != DRM_MODE_FLAG_NHSYNC) ||
	    (vflags != DRM_MODE_FLAG_PVSYNC && vflags != DRM_MODE_FLAG_NVSYNC) ||
	    (mode->flags & ~(DRM_MODE_FLAG_PHSYNC | DRM_MODE_FLAG_NHSYNC |
			     DRM_MODE_FLAG_PVSYNC | DRM_MODE_FLAG_NVSYNC)))
		return -EINVAL;
	hblank = mode->htotal - mode->hdisplay;
	vblank = mode->vtotal - mode->vdisplay;
	hoff = mode->hsync_start - mode->hdisplay;
	voff = mode->vsync_start - mode->vdisplay;
	hsync = mode->hsync_end - mode->hsync_start;
	vsync = mode->vsync_end - mode->vsync_start;
	if (hblank > 4095 || vblank > 4095 || hoff > 1023 || hsync > 1023 ||
	    voff > 63 || vsync > 63 || width > 4095 || height > 4095)
		return -EINVAL;

	timing->pixel_clock = cpu_to_le16(mode->clock / 10);
	pixel->hactive_lo = mode->hdisplay;
	pixel->hblank_lo = hblank;
	pixel->hactive_hblank_hi = (mode->hdisplay >> 8) << 4 | hblank >> 8;
	pixel->vactive_lo = mode->vdisplay;
	pixel->vblank_lo = vblank;
	pixel->vactive_vblank_hi = (mode->vdisplay >> 8) << 4 | vblank >> 8;
	pixel->hsync_offset_lo = hoff;
	pixel->hsync_pulse_width_lo = hsync;
	pixel->vsync_offset_pulse_width_lo = (voff & 15) << 4 | (vsync & 15);
	pixel->hsync_vsync_offset_pulse_width_hi = (hoff >> 8) << 6 | (hsync >> 8) << 4 |
						 (voff >> 4) << 2 | vsync >> 4;
	pixel->width_mm_lo = width;
	pixel->height_mm_lo = height;
	pixel->width_height_mm_hi = (width >> 8) << 4 | height >> 8;
	pixel->misc = DRM_EDID_PT_SEPARATE_SYNC;
	if (hflags == DRM_MODE_FLAG_PHSYNC)
		pixel->misc |= DRM_EDID_PT_HSYNC_POSITIVE;
	if (vflags == DRM_MODE_FLAG_PVSYNC)
		pixel->misc |= DRM_EDID_PT_VSYNC_POSITIVE;
	return 0;
}

static int dcp_metadata_text(struct detailed_timing *timing, u8 type, const char *text)
{
	size_t length = strnlen(text, 14);
	u8 *out = timing->data.other_data.data.str.str;

	if (!length || length > 13)
		return -EINVAL;
	for (size_t i = 0; i < length; i++)
		if (text[i] < 0x20 || text[i] > 0x7e)
			return -EINVAL;
	timing->data.other_data.type = type;
	memset(out, ' ', 13);
	memcpy(out, text, length);
	if (length < 13)
		out[length] = '\n';
	return 0;
}

static s64 dcp_metadata_side(const u32 *a, const u32 *b, const u32 *point)
{
	return ((s64)b[0] - a[0]) * ((s64)point[1] - a[1]) -
	       ((s64)b[1] - a[1]) * ((s64)point[0] - a[0]);
}

/* Build only a base block: the authoritative firmware mode list stays separate. */
static int dcp_metadata_block(struct edid *edid, const struct dcp_display_metadata *metadata,
			      const struct drm_display_mode *mode)
{
	u32 coordinate[8], manufacturer = 0, gamma;
	u8 checksum = 0;
	int ret;

	if (!metadata->width_cm || metadata->width_cm > 255 ||
	    !metadata->height_cm || metadata->height_cm > 255 ||
	    metadata->product > U16_MAX || metadata->week > 53 ||
	    metadata->year < 1990 || metadata->year > 2245 ||
	    metadata->gamma < 65536 || metadata->gamma > (354U << 16) / 100 ||
	    strnlen(metadata->manufacturer, 4) != 3)
		return -EINVAL;
	for (unsigned int i = 0; i < 3; i++) {
		if (metadata->manufacturer[i] < 'A' || metadata->manufacturer[i] > 'Z')
			return -EINVAL;
		manufacturer = manufacturer << 5 | (metadata->manufacturer[i] - '@');
	}
	for (unsigned int i = 0; i < 8; i++) {
		u32 value = i < 6 ? metadata->chromaticity[i / 2][i % 2] : metadata->white[i - 6];

		if (!value || value >= 65536)
			return -EINVAL;
		coordinate[i] = DIV_ROUND_CLOSEST(value, 64);
		if (!coordinate[i] || coordinate[i] > 1023)
			return -EINVAL;
		if ((i & 1) && coordinate[i - 1] + coordinate[i] > 1024)
			return -EINVAL;
	}
	/* Degenerate primaries or a white point outside their triangle are not usable. */
	{
		s64 area = dcp_metadata_side(coordinate, coordinate + 2, coordinate + 4);

		if (!area)
			return -EINVAL;
		for (unsigned int i = 0; i < 3; i++) {
			s64 side = dcp_metadata_side(coordinate + 2 * i,
				coordinate + 2 * ((i + 1) % 3), coordinate + 6);

			if ((area > 0 && side < 0) || (area < 0 && side > 0))
				return -EINVAL;
		}
	}
	gamma = DIV_ROUND_CLOSEST(metadata->gamma * 100, 65536) - 100;
	if (gamma > 254)
		return -EINVAL;

	memset(edid, 0, sizeof(*edid));
	memcpy(edid->header, "\x00\xff\xff\xff\xff\xff\xff\x00", 8);
	edid->product_id.manufacturer_name = cpu_to_be16(manufacturer);
	edid->product_id.product_code = cpu_to_le16(metadata->product);
	edid->product_id.serial_number = cpu_to_le32(metadata->serial);
	edid->product_id.week_of_manufacture = metadata->week;
	edid->product_id.year_of_manufacture = metadata->year - 1990;
	edid->version = 1;
	edid->revision = 4;
	/* No claim about connector interface, bit depth, HDR, audio or VRR. */
	edid->input = DRM_EDID_INPUT_DIGITAL;
	edid->width_cm = metadata->width_cm;
	edid->height_cm = metadata->height_cm;
	edid->gamma = gamma;
	edid->features = DRM_EDID_FEATURE_PREFERRED_TIMING | DRM_EDID_FEATURE_RGB;
	edid->red_green_lo = (coordinate[0] & 3) << 6 | (coordinate[1] & 3) << 4 |
			     (coordinate[2] & 3) << 2 | (coordinate[3] & 3);
	edid->blue_white_lo = (coordinate[4] & 3) << 6 | (coordinate[5] & 3) << 4 |
			      (coordinate[6] & 3) << 2 | (coordinate[7] & 3);
	edid->red_x = coordinate[0] >> 2;
	edid->red_y = coordinate[1] >> 2;
	edid->green_x = coordinate[2] >> 2;
	edid->green_y = coordinate[3] >> 2;
	edid->blue_x = coordinate[4] >> 2;
	edid->blue_y = coordinate[5] >> 2;
	edid->white_x = coordinate[6] >> 2;
	edid->white_y = coordinate[7] >> 2;
	memset(edid->standard_timings, 1, sizeof(edid->standard_timings));
	ret = dcp_metadata_timing(&edid->detailed_timings[0], mode, metadata);
	if (!ret)
		ret = dcp_metadata_text(&edid->detailed_timings[1], EDID_DETAIL_MONITOR_NAME,
					metadata->name);
	if (!ret)
		ret = dcp_metadata_text(&edid->detailed_timings[2], EDID_DETAIL_MONITOR_STRING,
					"DCP metadata");
	if (!ret)
		ret = dcp_metadata_text(&edid->detailed_timings[3],
					metadata->serial_text[0] ? EDID_DETAIL_MONITOR_SERIAL :
					EDID_DETAIL_MONITOR_STRING,
					metadata->serial_text[0] ?
					metadata->serial_text : "FW derived");
	if (ret)
		return ret;
	for (unsigned int i = 0; i < sizeof(*edid) - 1; i++)
		checksum += ((const u8 *)edid)[i];
	edid->checksum = -checksum;
	return 0;
}

const struct drm_edid *dcp_metadata_edid(const struct dcp_display_metadata *metadata,
					 const struct drm_display_mode *mode)
{
	struct edid block;
	const struct drm_edid *edid;
	int ret = dcp_metadata_block(&block, metadata, mode);

	if (ret)
		return ERR_PTR(ret);
	edid = drm_edid_alloc(&block, sizeof(block));
	if (!edid)
		return ERR_PTR(-ENOMEM);
	if (!drm_edid_valid(edid)) {
		drm_edid_free(edid);
		return ERR_PTR(-EINVAL);
	}
	return edid;
}
