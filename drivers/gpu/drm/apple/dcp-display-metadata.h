/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
#ifndef APPLE_DCP_DISPLAY_METADATA_H
#define APPLE_DCP_DISPLAY_METADATA_H

#include <linux/types.h>

/* Values reported by DisplayAttributes, not a copy of the sink's EDID. */
struct dcp_display_metadata {
	u32 width_cm, height_cm;
	u32 chromaticity[3][2];
	u32 white[2];
	u32 gamma;
	u32 product, serial, week, year;
	char manufacturer[4];
	char name[14];
	char serial_text[14];
};

struct drm_edid;
struct drm_display_mode;
const struct drm_edid *dcp_metadata_edid(const struct dcp_display_metadata *metadata,
					 const struct drm_display_mode *mode);

#endif
