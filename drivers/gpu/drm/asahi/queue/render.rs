// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(clippy::unusual_byte_groupings)]

//! Render work queue.
//!
//! A render queue consists of two underlying WorkQueues, one for vertex and one for fragment work.
//! This module is in charge of creating all of the firmware structures required to submit 3D
//! rendering work to the GPU, based on the userspace command buffer.

use super::common;
use crate::alloc::Allocator;
use crate::debug::*;
use crate::fw::types::*;
use crate::gpu::GpuManager;
use crate::util::*;
use crate::{
    buffer,
    file,
    fw,
    gpu,
    microseq, //
};
use crate::{
    inner_ptr,
    inner_weak_ptr, //
};
use core::sync::atomic::Ordering;
use kernel::dma_fence::RawDmaFence;
use kernel::drm::sched::Job;
use kernel::prelude::*;
use kernel::sync::Arc;
use kernel::uapi;
use kernel::xarray;

const DEBUG_CLASS: DebugFlags = DebugFlags::Render;

/// Tiling/Vertex control bit to disable using more than one GPU cluster. This results in decreased
/// throughput but also less latency, which is probably desirable for light vertex loads where the
/// overhead of clustering/merging would exceed the time it takes to just run the job on one
/// cluster.
const TILECTL_DISABLE_CLUSTERING: u32 = 1u32 << 0;

// Apple G15 uses the same raw Render mode selector in TA and Fragment
// register generation. Mode 2 suppresses the contribution, mode 3 selects
// the lower value, and every other byte value selects the default/high value.
// Keep the raw byte explicit until its userspace semantic is recovered.
const fn g15_render_mode_selector(mode: u8, lower: u64, default: u64) -> u64 {
    if mode == 2 {
        0
    } else if mode == 3 {
        lower
    } else {
        default
    }
}

// Exact Apple G15 value for hardware register 0x10039. RTM +0x98 is the
// layer-count dimension; raw Render +0x618 is the process-empty-tiles byte;
// raw +0x619/+0x61a are unresolved cross-engine mode selectors.
const fn g15_fragment_tile_config(
    layers: u32,
    process_empty_tiles: bool,
    raw_mode_619: u8,
    raw_mode_61a: u8,
) -> u64 {
    (if layers > 1 { 1 } else { 0 })
        | (if process_empty_tiles { 0x1_0000 } else { 0 })
        | g15_render_mode_selector(raw_mode_619, 0x100, 0x200)
        | g15_render_mode_selector(raw_mode_61a, 0x40, 0x80)
}

// Linux has historically exposed only the normal/default tiler mode (`unk1=false`).
// On G14X that means TA 0x10141 = 0x200 and Fragment 0x10039 contributes
// 0x280.  Exact G15 splits those contributions across raw +0x619/+0x61a;
// their default/high branches reproduce the same Linux-visible state.  Modes 2/3
// are private Apple extensions and need a future UAPI semantic before Linux can
// request them.
const G15_LINUX_TILER_MODE_10141: u64 = 0x200;

// G15 Fragment 0x1a0a9 (and TA 0x1a099) comes from raw Render +0x638.
// Apple packs an initial-render "any attachment loadAction=CLEAR" boolean in
// byte 0 and native in-render resolve state in byte 1. Honeykrisp performs
// resolves as a separate control stream, so byte 1 is zero for Linux. The
// remaining producer is an explicit has-load-clear semantic that the current
// render UAPI does not expose; keep only the exact formula here until it does.
const fn g15_render_clear_state(has_load_clear: bool) -> u64 {
    if has_load_clear { 1 } else { 0 }
}

const _: [(); 0] = [(); g15_render_clear_state(false) as usize];
const _: [(); 1] = [(); g15_render_clear_state(true) as usize];

const fn g15_fragment_tile_config_linux(layers: u32, process_empty_tiles: bool) -> u64 {
    (if layers > 1 { 1 } else { 0 })
        | (if process_empty_tiles { 0x1_0000 } else { 0 })
        | 0x280
}

// Exact J615/G15G C0 value formula for hardware register 0x16068, with
// no unresolved accelerator state: base configureDevice() clears accelerator
// +0x650 bit 21 with mask 0xfffffe0c2287 and G15/G15G never set it again, so
// Apple's result-bit-20 accelerator contribution is exactly zero here.
const fn g15_fragment_tilecfg(
    utile_config: u32,
    layers: u32,
    te_screen: u32,
    raw_mode_619: u8,
) -> u64 {
    ((utile_config as u64 & 0xf000) << 28)
        | ((if layers > 1 { 1u64 } else { 0 }) << 32)
        | (g15_render_mode_selector(raw_mode_619, 0x100, 0x200) << 28)
        | ((te_screen as u64 & 0x1ff) << 44)
        | ((te_screen as u64 & 0x1ff000) << 41)
        | 0x3617f
}

const fn g15_fragment_tilecfg_linux(utile_config: u32, layers: u32, te_screen: u32) -> u64 {
    ((utile_config as u64 & 0xf000) << 28)
        | ((if layers > 1 { 1u64 } else { 0 }) << 32)
        | 0x20_00000000
        | ((te_screen as u64 & 0x1ff) << 44)
        | ((te_screen as u64 & 0x1ff000) << 41)
        | 0x3617f
}

// G15 TA 0x10169 is layer-count state plus the raw +0x618 process-empty-tiles
// mode. Apple then derives 0x1c9e8 by masking this value with 0x47ff.
const fn g15_ta_render_target_max(layers: u32, process_empty_tiles: bool) -> u32 {
    let mut value = (layers - 1) | 0x8000;
    if layers > 1 {
        value |= 0x4000 | if process_empty_tiles { 0x2000 } else { 0x1000 };
    }
    value
}

const _: [(); 0x8000] = [(); g15_ta_render_target_max(1, false) as usize];
const _: [(); 0xd001] = [(); g15_ta_render_target_max(2, false) as usize];
const _: [(); 0xe001] = [(); g15_ta_render_target_max(2, true) as usize];

// Default/zero raw mode reduces 0x10039 to Linux's historical tile_config
// composition. These constants also pin the non-default selector branches.
const _: [(); 0x280] = [(); g15_fragment_tile_config(1, false, 0, 0) as usize];
const _: [(); 0x10281] = [(); g15_fragment_tile_config(2, true, 0, 0) as usize];
const _: [(); 0x280] = [(); g15_fragment_tile_config_linux(1, false) as usize];
const _: [(); 0x10281] = [(); g15_fragment_tile_config_linux(2, true) as usize];
const _: [(); 0] = [(); g15_fragment_tile_config(1, false, 2, 2) as usize];
const _: [(); 0x140] = [(); g15_fragment_tile_config(1, false, 3, 3) as usize];

// For a 32x32 utile (utile_config high bits 0xa000), one layer and zero tile
// counts, raw mode 0 supplies the same historical 0x20_00000000 mode bit,
// while G15 changes the fixed low constant from 0x36011 to exact 0x3617f.
const _: [(); 0xa200003617f] =
    [(); g15_fragment_tilecfg(0xa000, 1, 0, 0) as usize];
const _: [(); 0xa200003617f] =
    [(); g15_fragment_tilecfg_linux(0xa000, 1, 0) as usize];

#[versions(AGX)]
impl super::QueueInner::ver {
    /// Get the appropriate tiling parameters for a given userspace command buffer.
    fn get_tiling_params(
        cmdbuf: &uapi::drm_asahi_cmd_render,
        num_clusters: u32,
    ) -> Result<buffer::TileInfo> {
        let width: u32 = cmdbuf.width_px as u32;
        let height: u32 = cmdbuf.height_px as u32;
        let layers: u32 = cmdbuf.layers as u32;

        if layers == 0 || layers > 2048 {
            cls_pr_debug!(Errors, "Layer count invalid ({})\n", layers);
            return Err(EINVAL);
        }

        // This is overflow safe: all these calculations are done in u32.
        // At 64Kx64K max dimensions above, this is 2**32 pixels max.
        // In terms of tiles that are always larger than one pixel,
        // this can never overflow. Note that real actual dimensions
        // are limited to 16K * 16K below anyway.
        //
        // Once we multiply by the layer count, then we need to check
        // for overflow or use u64.

        let tile_width = 32u32;
        let tile_height = 32u32;

        let utile_width = cmdbuf.utile_width_px as u32;
        let utile_height = cmdbuf.utile_height_px as u32;

        match (utile_width, utile_height) {
            (32, 32) | (32, 16) | (16, 16) => (),
            _ => {
                cls_pr_debug!(
                    Errors,
                    "uTile size invalid ({} x {})\n",
                    utile_width,
                    utile_height
                );
                return Err(EINVAL);
            }
        };

        let utiles_per_tile_x = tile_width / utile_width;
        let utiles_per_tile_y = tile_height / utile_height;

        let utiles_per_tile = utiles_per_tile_x * utiles_per_tile_y;

        let tiles_x = width.div_ceil(tile_width);
        let tiles_y = height.div_ceil(tile_height);
        let tiles = tiles_x * tiles_y;

        let mtiles_x = 4u32;
        let mtiles_y = 4u32;
        let mtiles = mtiles_x * mtiles_y;

        let tiles_per_mtile_x = align(tiles_x.div_ceil(mtiles_x), 4);
        let tiles_per_mtile_y = align(tiles_y.div_ceil(mtiles_y), 4);
        let tiles_per_mtile = tiles_per_mtile_x * tiles_per_mtile_y;

        let mtile_x1 = tiles_per_mtile_x;
        let mtile_x2 = 2 * tiles_per_mtile_x;
        let mtile_x3 = 3 * tiles_per_mtile_x;

        let mtile_y1 = tiles_per_mtile_y;
        let mtile_y2 = 2 * tiles_per_mtile_y;
        let mtile_y3 = 3 * tiles_per_mtile_y;

        let rgn_entry_size = 5;
        // Macrotile stride in 32-bit words
        let rgn_size = align(rgn_entry_size * tiles_per_mtile * utiles_per_tile, 4) / 4;
        let tilemap_size = (4 * rgn_size * mtiles) as usize * layers as usize;

        let tpc_entry_size = 8;
        // TPC stride in 32-bit words
        let tpc_mtile_stride = tpc_entry_size * utiles_per_tile * tiles_per_mtile / 4;
        let tpc_size =
            (4 * tpc_mtile_stride * mtiles) as usize * layers as usize * num_clusters as usize;

        // No idea where this comes from, but it fits what macOS does...
        // GUESS: Number of 32K heap blocks to fit a 5-byte region header/pointer per tile?
        // That would make a ton of sense...
        let meta1_layer_stride = if num_clusters > 1 {
            (align(tiles_x, 2) * align(tiles_y, 4) * utiles_per_tile).div_ceil(0x1980)
        } else {
            0
        };

        let mut min_tvb_blocks = align((tiles_x * tiles_y).div_ceil(128), 8);

        if num_clusters > 1 {
            min_tvb_blocks = min_tvb_blocks.max(7 + 2 * layers);
        }

        Ok(buffer::TileInfo {
            tiles_x,
            tiles_y,
            tiles,
            utile_width,
            utile_height,
            //mtiles_x,
            //mtiles_y,
            tiles_per_mtile_x,
            tiles_per_mtile_y,
            //tiles_per_mtile,
            utiles_per_mtile_x: tiles_per_mtile_x * utiles_per_tile_x,
            utiles_per_mtile_y: tiles_per_mtile_y * utiles_per_tile_y,
            //utiles_per_mtile: tiles_per_mtile * utiles_per_tile,
            tilemap_size,
            tpc_size,
            meta1_layer_stride,
            #[ver(G < G14X)]
            meta1_blocks: meta1_layer_stride * (cmdbuf.layers as u32),
            #[ver(G >= G14X)]
            meta1_blocks: meta1_layer_stride,
            layermeta_size: if layers > 1 { 0x100 } else { 0 },
            min_tvb_blocks: min_tvb_blocks as usize,
            params: fw::vertex::raw::TilingParameters {
                rgn_size,
                unk_4: 0x88,
                ppp_ctrl: cmdbuf.ppp_ctrl,
                x_max: (width - 1) as u16,
                y_max: (height - 1) as u16,
                te_screen: ((tiles_y - 1) << 12) | (tiles_x - 1),
                te_mtile1: mtile_x3 | (mtile_x2 << 9) | (mtile_x1 << 18),
                te_mtile2: mtile_y3 | (mtile_y2 << 9) | (mtile_y1 << 18),
                tiles_per_mtile,
                tpc_stride: tpc_mtile_stride,
                unk_24: 0x100,
                unk_28: if layers > 1 {
                    0xe000 | (layers - 1)
                } else {
                    0x8000
                },
                helper_cfg: cmdbuf.vertex_helper.cfg,
                __pad: Default::default(),
            },
        })
    }

    /// Submit work to a render queue.
    pub(super) fn submit_render(
        &self,
        job: &mut Job<super::QueueJob::ver>,
        cmdbuf: &uapi::drm_asahi_cmd_render,
        vertex_attachments: &microseq::Attachments,
        fragment_attachments: &microseq::Attachments,
        objects: Pin<&xarray::XArray<KBox<file::Object>>>,
        id: u64,
        flush_stamps: bool,
    ) -> Result {
        mod_dev_dbg!(self.dev, "[Submission {}] Render!\n", id);

        if cmdbuf.flags
            & !(uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_VERTEX_SCRATCH
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING
                | uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_DBIAS_IS_INT) as u32
            != 0
        {
            cls_pr_debug!(Errors, "Invalid flags ({:#x})\n", cmdbuf.flags);
            return Err(EINVAL);
        }

        if cmdbuf.width_px == 0
            || cmdbuf.height_px == 0
            || cmdbuf.width_px > 16384
            || cmdbuf.height_px > 16384
        {
            cls_pr_debug!(
                Errors,
                "Invalid dimensions ({}x{})\n",
                cmdbuf.width_px,
                cmdbuf.height_px
            );
            return Err(EINVAL);
        }

        let mut vtx_user_timestamps: fw::job::UserTimestamps = Default::default();
        let mut frg_user_timestamps: fw::job::UserTimestamps = Default::default();

        vtx_user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts_vtx.start)?;
        vtx_user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts_vtx.end)?;
        frg_user_timestamps.start = common::get_timestamp_object(objects, cmdbuf.ts_frag.start)?;
        frg_user_timestamps.end = common::get_timestamp_object(objects, cmdbuf.ts_frag.end)?;

        let gpu = match (*self.dev)
            .gpu
            .as_any()
            .downcast_ref::<gpu::GpuManager::ver>()
        {
            Some(gpu) => gpu,
            None => {
                dev_crit!(self.dev.as_ref(), "GpuManager mismatched with Queue!\n");
                return Err(EIO);
            }
        };

        let nclusters = gpu.get_dyncfg().id.num_clusters;

        // Can be set to false to disable clustering (for simpler jobs), but then the
        // core masks below should be adjusted to cover a single rolling cluster.
        let mut clustering = nclusters > 1;

        if debug_enabled(debug::DebugFlags::DisableClustering)
            || cmdbuf.flags
                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING as u32
                != 0
        {
            clustering = false;
        }

        #[ver(G != G14 && G != G15)]
        let tiling_control = {
            let render_cfg = gpu.get_cfg().render;
            let mut tiling_control = render_cfg.tiling_control;

            if !clustering {
                tiling_control |= TILECTL_DISABLE_CLUSTERING;
            }
            tiling_control
        };

        let mut alloc = gpu.alloc();
        let kalloc = &mut *alloc;

        // This sequence number increases per new client/VM? assigned to some slot,
        // but it's unclear *which* slot...
        let slot_client_seq: u8 = (self.id & 0xff) as u8;
        #[ver(G == G15)]
        let _ = slot_client_seq;

        let tile_info = Self::get_tiling_params(&cmdbuf, if clustering { nclusters } else { 1 })?;

        let buffer = &self.buffer;
        let notifier = self.notifier.clone();

        let tvb_autogrown = buffer.auto_grow()?;
        if tvb_autogrown {
            let new_size = buffer.block_count() as usize;
            cls_dev_dbg!(
                TVBStats,
                &self.dev,
                "[Submission {}] TVB grew to {} bytes ({} blocks) due to overflows\n",
                id,
                new_size * buffer::BLOCK_SIZE,
                new_size,
            );
        }

        let tvb_grown = buffer.ensure_blocks(tile_info.min_tvb_blocks)?;
        if tvb_grown {
            cls_dev_dbg!(
                TVBStats,
                &self.dev,
                "[Submission {}] TVB grew to {} bytes ({} blocks) due to dimensions ({}x{})\n",
                id,
                tile_info.min_tvb_blocks * buffer::BLOCK_SIZE,
                tile_info.min_tvb_blocks,
                cmdbuf.width_px,
                cmdbuf.height_px
            );
        }

        let scene = Arc::new(buffer.new_scene(kalloc, &tile_info)?, GFP_KERNEL)?;

        // Apple AGXCommandQueue::processRenderSetup() advances the selected
        // AGXParameterManagement +0x2c record index after command validation and
        // before descriptor loading. The PM object is AGX3DWorkQueue-owned, which
        // matches this QueueInner lifetime. The range-5/range-7 PM resources are
        // now reconstructed; only G15 RegisterArray emission remains disabled.
        #[ver(G == G15)]
        let g15_pm_record_index = {
            let current = self
                .g15_pm_record_index
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(buffer::g15_j615_pm_next_record_index(current))
                })
                .expect("G15 PM record update is unconditional");
            buffer::g15_j615_pm_next_record_index(current)
        };
        #[ver(G == G15)]
        let g15_pm_scene_slice_offset =
            buffer::g15_j615_pm_scene_slice_offset(g15_pm_record_index);
        #[ver(G == G15)]
        let g15_pm_scene_slice_gpuva: u64 = self
            .g15_pm_scene_alloc
            .gpu_offset_pointer(g15_pm_scene_slice_offset)
            .into();
        #[ver(G == G15)]
        let g15_pm_record_gpuva: u64 = self
            .g15_pm_records
            .gpu_offset_pointer(g15_pm_record_index as usize)
            .into();
        #[ver(G == G15)]
        let g15_pm_page_metrics_gpuva: u64 = self
            ._g15_pm_page_metrics
            .gpu_offset_pointer(buffer::g15_j615_pm_page_metrics_slot_offset(
                g15_pm_record_index,
            ))
            .into();

        let vm_bind = job.vm_bind.clone();

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] VM slot = {}\n",
            id,
            vm_bind.slot()
        );

        let ev_vtx = job.get_vtx()?.event_info();
        let ev_frag = job.get_frag()?.event_info();

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Vert event #{} -> {:#x?}\n",
            id,
            ev_vtx.slot,
            ev_vtx.value.next(),
        );
        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Frag event #{} -> {:#x?}\n",
            id,
            ev_frag.slot,
            ev_frag.value.next(),
        );

        #[ver(G != G15)]
        let uuid_3d = 0;
        #[ver(G != G15)]
        let uuid_ta = 0;
        // Apple allocates the 3D descriptor before the TA descriptor and both
        // consume the same global command-descriptor ID sequence.
        #[ver(G == G15)]
        let uuid_3d = common::next_g15_command_uuid();
        #[ver(G == G15)]
        let uuid_ta = common::next_g15_command_uuid();

        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Vert UUID = {:#x?}\n",
            id,
            uuid_ta
        );
        mod_dev_dbg!(
            self.dev,
            "[Submission {}] Frag UUID = {:#x?}\n",
            id,
            uuid_3d
        );

        let fence = job.fence.clone();
        let frag_job = job.get_frag()?;

        mod_dev_dbg!(self.dev, "[Submission {}] Create Barrier\n", id);
        let barrier = kalloc.private.new_init(
            pin_init::zeroed::<fw::workqueue::Barrier::ver>(),
            |_inner, _p| {
                try_init!(fw::workqueue::raw::Barrier::ver {
                    tag: fw::workqueue::CommandType::Barrier,
                    wait_stamp: ev_vtx.fw_stamp_pointer,
                    #[ver(G == G15)]
                    wait_stamp_2: ev_vtx.fw_stamp_pointer,
                    wait_value: ev_vtx.value.next(),
                    wait_slot: ev_vtx.slot,
                    stamp_self: ev_frag.value.next(),
                    uuid: uuid_3d,
                    external_barrier: 0,
                    internal_barrier_type: 0,
                    padding: Default::default(),
                })
            },
        )?;

        mod_dev_dbg!(self.dev, "[Submission {}] Add Barrier\n", id);
        frag_job.add(barrier, vm_bind.slot())?;

        let timestamps = Arc::new(
            kalloc.shared.new_default::<fw::job::RenderTimestamps>()?,
            GFP_KERNEL,
        )?;

        let unk1 = false;

        #[ver(G != G15)]
        let tile_config: u64 = {
            let mut tile_config = 0;
            if !unk1 {
                tile_config |= 0x280;
            }
            if cmdbuf.layers > 1 {
                tile_config |= 1;
            }
            if cmdbuf.flags
                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES as u32
                != 0
            {
                tile_config |= 0x10000;
            }
            tile_config
        };

        let samples_log2 = match cmdbuf.samples {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => {
                cls_pr_debug!(Errors, "Invalid sample count {}\n", cmdbuf.samples);
                return Err(EINVAL);
            }
        };

        let utile_config = ((tile_info.utile_width / 16) << 12)
            | ((tile_info.utile_height / 16) << 14)
            | samples_log2;

        // Calculate the number of 2KiB blocks to allocate per utile. This is
        // just a bit of dimensional analysis.
        #[ver(G != G15)]
        let pixels_per_utile: u32 =
            (cmdbuf.utile_width_px as u32) * (cmdbuf.utile_height_px as u32);
        #[ver(G != G15)]
        let samples_per_utile: u32 = pixels_per_utile << samples_log2;
        #[ver(G != G15)]
        let utile_size_bytes: u32 = (cmdbuf.sample_size_B as u32) * samples_per_utile;
        #[ver(G != G15)]
        let block_size_bytes: u32 = 2048;
        #[ver(G != G15)]
        let blocks_per_utile: u32 = utile_size_bytes.div_ceil(block_size_bytes);

        #[ver(G >= G14X)]
        let frg_tilecfg = 0x0000000_00036011
            | (((tile_info.tiles_x - 1) as u64) << 44)
            | (((tile_info.tiles_y - 1) as u64) << 53)
            | (if unk1 { 0 } else { 0x20_00000000 })
            | (if cmdbuf.layers > 1 { 0x1_00000000 } else { 0 })
            | ((utile_config as u64 & 0xf000) << 28);

        // TODO: check
        // Normal Apple G15 AGX3DWorkQueue::submitCommand() invokes the 3D
        // channel submitBuffer() first and the TA channel submitBuffer() second.
        // Both consume AGXCommandQueue +0x5c8, so reserve Fragment n then TA n+1.
        #[ver(V >= V13_0B4)]
        let count_frag = self.counter.fetch_add(2, Ordering::Relaxed);
        #[ver(V >= V13_0B4)]
        let count_vtx = count_frag + 1;

        // Unknowns handling

        #[ver(G >= G14 && G != G15)]
        let g14_unk = 0x4040404;
        #[ver(G < G14)]
        let g14_unk = 0;
        #[ver(G < G14X && G != G15)]
        let frg_unk_140 = 0x8c60;
        let frg_unk_158 = 0x1c;
        #[ver(G >= G14)]
        let load_bgobjvals = cmdbuf.isp_bgobjvals as u64;
        #[ver(G < G14)]
        let load_bgobjvals = cmdbuf.isp_bgobjvals as u64 | 0x400;
        #[ver(G != G15)]
        let reload_zlsctrl = cmdbuf.zls_ctrl;
        let iogpu_unk54: u64 = 0x3a0012006b0003;
        let iogpu_unk56: u64 = 1;
        #[ver(G < G14)]
        let tiling_control_2 = 0;
        #[ver(G >= G14X)]
        let tiling_control_2 = 4;
        #[ver(G >= G14X)]
        let vtx_unk_f0 = 0x1c;
        #[ver(G < G14)]
        let vtx_unk_f0 = 0x1c + (align(tile_info.meta1_blocks, 4) as u64);
        let vtx_unk_118: u64 = 0x1c;

        // DRM_ASAHI_RENDER_DBIAS_IS_INT chosen to match hardware bit.
        #[ver(G != G15)]
        let isp_ctl = 0xc000u32
            | (cmdbuf.flags & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_DBIAS_IS_INT as u32);

        // Always allow preemption at the UAPI level
        #[ver(G != G15)]
        let no_preemption = false;

        mod_dev_dbg!(self.dev, "[Submission {}] Create Frag\n", id);
        let frag = GpuObject::new_init_prealloc(
            kalloc.gpu_ro.alloc_object()?,
            |ptr: GpuWeakPointer<fw::fragment::RunFragment::ver>| {
                let scene = scene.clone();
                let notifier = notifier.clone();
                let vm_bind = vm_bind.clone();
                let timestamps = timestamps.clone();
                let private = &mut kalloc.private;
                try_init!(fw::fragment::RunFragment::ver {
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = inner_weak_ptr!(
                            gpu.initdata.runtime_pointers.stats.frag.weak_pointer(),
                            stats
                        );

                        let start_frag = builder.add(microseq::StartFragment::ver {
                            header: microseq::op::StartFragment::HEADER,
                            #[ver(G < G14X && G != G15)]
                            job_params2: Some(inner_weak_ptr!(ptr, job_params2)),
                            #[ver(G < G14X && G != G15)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X || G == G15)]
                            job_params1: None,
                            #[ver(G >= G14X || G == G15)]
                            job_params2: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            scene: scene.gpu_pointer(),
                            stats,
                            busy_flag: inner_weak_ptr!(ptr, busy_flag),
                            tvb_overflow_count: inner_weak_ptr!(ptr, tvb_overflow_count),
                            #[ver(G != G15)]
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(G == G15)]
                            unk_pointer: inner_weak_ptr!(ptr, g15_raw_render_4b6_nonzero_ba0),
                            work_queue: ev_frag.info_ptr,
                            work_item: ptr,
                            vm_slot: vm_bind.slot(),
                            unk_50: 0x1, // fixed
                            event_generation: self.id as u32,
                            buffer_slot: scene.slot(),
                            sync_grow: 0,
                            event_seq: U64(ev_frag.event_seq),
                            unk_68: 0,
                            #[ver(G != G15)]
                            unk_758_flag: inner_weak_ptr!(ptr, unk_758_flag),
                            #[ver(G == G15)]
                            unk_758_flag: inner_weak_ptr!(ptr, g15_zero_a48),
                            #[ver(G != G15)]
                            unk_job_buf: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(G == G15)]
                            unk_job_buf: inner_weak_ptr!(ptr, g15_zero_bd8),
                            #[ver(V >= V13_3)]
                            unk_7c_0: U64(0),
                            unk_7c: 0,
                            unk_80: 0,
                            unk_84: unk1.into(),
                            uuid: uuid_3d,
                            attachments: *fragment_attachments,
                            padding: 0,
                            #[ver(V >= V13_0B4)]
                            counter: U64(count_frag),
                            #[ver(V >= V13_0B4)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                        })?;

                        #[ver(G != G15)]
                        if frg_user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_frag.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_3d,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Fragment),
                        })?;
                        #[ver(G >= G14X)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;

                        #[ver(G != G15)]
                        if frg_user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_frag.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_3d,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_frag);
                        builder.add(microseq::FinalizeFragment::ver {
                            header: microseq::op::FinalizeFragment::HEADER,
                            uuid: uuid_3d,
                            unk_8: 0,
                            fw_stamp: ev_frag.fw_stamp_pointer,
                            stamp_value: ev_frag.value.next(),
                            unk_18: 0,
                            scene: scene.weak_pointer(),
                            buffer: scene.weak_buffer_pointer(),
                            unk_2c: U64(1),
                            stats,
                            #[ver(G != G15)]
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(G == G15)]
                            unk_pointer: inner_weak_ptr!(ptr, g15_raw_render_4b6_nonzero_ba0),
                            busy_flag: inner_weak_ptr!(ptr, busy_flag),
                            work_queue: ev_frag.info_ptr,
                            work_item: ptr,
                            vm_slot: vm_bind.slot(),
                            unk_60: 0,
                            #[ver(G != G15)]
                            unk_758_flag: inner_weak_ptr!(ptr, unk_758_flag),
                            #[ver(G == G15)]
                            unk_758_flag: inner_weak_ptr!(ptr, g15_zero_a48),
                            #[ver(V >= V13_3)]
                            unk_6c_0: U64(0),
                            unk_6c: U64(0),
                            unk_74: U64(0),
                            unk_7c: U64(0),
                            unk_84: U64(0),
                            unk_8c: U64(0),
                            #[ver(G == G14 && V < V13_0B4)]
                            unk_8c_g14: U64(0),
                            restart_branch_offset: off,
                            has_attachments: (fragment_attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4)]
                            unk_9c: Default::default(),
                        })?;

                        builder.add(microseq::RetireStamp {
                            header: microseq::op::RetireStamp::HEADER,
                        })?;

                        builder.build(private)?
                    },
                    notifier,
                    scene,
                    vm_bind,
                    aux_fb: self.ualloc.lock().array_empty_tagged(0x8000, b"AXFB")?,
                    timestamps,
                    user_timestamps: frg_user_timestamps,
                })
            },
            |inner, _ptr| {
                #[ver(G != G15)]
                let vm_slot = vm_bind.slot();
                #[ver(G != G15)]
                let aux_fb_info = fw::fragment::raw::AuxFBInfo::ver {
                    isp_ctl: isp_ctl,
                    unk2: 0,
                    width: cmdbuf.width_px as u32,
                    height: cmdbuf.height_px as u32,
                    #[ver(V >= V13_0B4)]
                    unk3: U64(0x100000),
                };

                try_init!(fw::fragment::raw::RunFragment::ver {
                    tag: fw::workqueue::CommandType::RunFragment,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count_frag),
                    #[ver(G != G15)]
                    vm_slot,
                    #[ver(G == G15)]
                    g15_context_id_c: 0,
                    #[ver(G != G15)]
                    unk_8: 0,
                    #[ver(G != G15)]
                    microsequence: inner.micro_seq.gpu_pointer(),
                    #[ver(G != G15)]
                    microsequence_size: inner.micro_seq.len() as u32,
                    #[ver(G == G15)]
                    g15_pad_10: Default::default(),
                    #[ver(G != G15)]
                    notifier: inner.notifier.gpu_pointer(),
                    #[ver(G != G15)]
                    buffer: inner.scene.buffer_pointer(),
                    #[ver(G != G15)]
                    scene: inner.scene.gpu_pointer(),
                    #[ver(G != G15)]
                    unk_buffer_buf: inner.scene.kernel_buffer_pointer(),
                    #[ver(G == G15)]
                    g15_cmd_buffer_state_398_20: U64(0),
                    #[ver(G == G15)]
                    g15_buffer_fwva_28: U64(inner.scene.buffer_pointer().into()),
                    #[ver(G == G15)]
                    g15_pm_record_fwva_30: U64(g15_pm_record_gpuva),
                    #[ver(G == G15)]
                    g15_pm_state_fwva_38: U64(0),
                    #[ver(G != G15)]
                    tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                    #[ver(G != G15)]
                    ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl),
                    #[ver(G != G15)]
                    samples: cmdbuf.samples as u32,
                    #[ver(G != G15)]
                    tiles_per_mtile_y: tile_info.tiles_per_mtile_y as u16,
                    #[ver(G != G15)]
                    tiles_per_mtile_x: tile_info.tiles_per_mtile_x as u16,
                    #[ver(G != G15)]
                    unk_50: U64(0),
                    #[ver(G != G15)]
                    unk_58: U64(0),
                    #[ver(G != G15)]
                    isp_merge_upper_x: F32::from_bits(cmdbuf.isp_merge_upper_x),
                    #[ver(G != G15)]
                    isp_merge_upper_y: F32::from_bits(cmdbuf.isp_merge_upper_y),
                    #[ver(G != G15)]
                    unk_68: U64(0),
                    #[ver(G != G15)]
                    tile_count: U64(tile_info.tiles as u64),
                    #[ver(G == G15)]
                    g15_rtm_298_addr_40: U64(0),
                    #[ver(G == G15)]
                    g15_rtm_b0_48: U64(0),
                    #[ver(G == G15)]
                    g15_rtm_b8_50: 0,
                    #[ver(G == G15)]
                    g15_rtm_bc_54: 0,
                    #[ver(G == G15)]
                    g15_rtm_20_low_58: 0,
                    #[ver(G == G15)]
                    g15_pad_5c: 0,
                    #[ver(G == G15)]
                    g15_rtm_c0_cc_60: Default::default(),
                    #[ver(G == G15)]
                    g15_rtm_d0_d4_70: Default::default(),
                    #[ver(G == G15)]
                    g15_rtm_118_78: 0,
                    #[ver(G == G15)]
                    g15_pad_7c: 0,
                    #[ver(G == G15)]
                    _g15_lifetime: core::marker::PhantomData,
                    #[ver(G < G14X && G != G15)]
                    job_params1 <- try_init!(fw::fragment::raw::JobParameters1::ver {
                        utile_config,
                        unk_4: 0,
                        bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.bg.rsrc_spec as u64),
                            address: U64(cmdbuf.bg.usc as u64),
                        },
                        ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl),
                        isp_scissor_base: U64(cmdbuf.isp_scissor_base),
                        isp_dbias_base: U64(cmdbuf.isp_dbias_base),
                        isp_oclqry_base: U64(cmdbuf.isp_oclqry_base),
                        aux_fb_info,
                        isp_zls_pixels: U64(cmdbuf.isp_zls_pixels as u64),
                        zls_ctrl: U64(cmdbuf.zls_ctrl),
                        #[ver(G >= G14)]
                        unk_58_g14_0: U64(g14_unk),
                        #[ver(G >= G14)]
                        unk_58_g14_8: U64(0),
                        z_load: U64(cmdbuf.depth.base),
                        z_store: U64(cmdbuf.depth.base),
                        s_load: U64(cmdbuf.stencil.base),
                        s_store: U64(cmdbuf.stencil.base),
                        #[ver(G >= G14)]
                        unk_68_g14_0: Default::default(),
                        z_load_stride: U64(cmdbuf.depth.stride as u64),
                        z_store_stride: U64(cmdbuf.depth.stride as u64),
                        s_load_stride: U64(cmdbuf.stencil.stride as u64),
                        s_store_stride: U64(cmdbuf.stencil.stride as u64),
                        z_load_comp: U64(cmdbuf.depth.comp_base),
                        z_load_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        z_store_comp: U64(cmdbuf.depth.comp_base),
                        z_store_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        s_load_comp: U64(cmdbuf.stencil.comp_base),
                        s_load_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        s_store_comp: U64(cmdbuf.stencil.comp_base),
                        s_store_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                        tvb_layermeta: inner.scene.tvb_layermeta_pointer(),
                        mtile_stride_dwords: U64((4 * tile_info.params.rgn_size as u64) << 24),
                        tvb_heapmeta: inner.scene.tvb_heapmeta_pointer(),
                        tile_config: U64(tile_config),
                        aux_fb: inner.aux_fb.gpu_pointer(),
                        unk_108: Default::default(),
                        usc_exec_base_isp: U64(self.usc_exec_base),
                        unk_140: U64(frg_unk_140),
                        helper_program: cmdbuf.fragment_helper.binary,
                        unk_14c: 0,
                        helper_arg: U64(cmdbuf.fragment_helper.data),
                        unk_158: U64(frg_unk_158),
                        unk_160: U64(0),
                        __pad: Default::default(),
                        #[ver(V < V13_0B4)]
                        __pad1: Default::default(),
                    }),
                    #[ver(G < G14X && G != G15)]
                    job_params2 <- try_init!(fw::fragment::raw::JobParameters2 {
                        eot_rsrc_spec: cmdbuf.eot.rsrc_spec,
                        eot_usc: cmdbuf.eot.usc,
                        unk_8: 0x0,
                        unk_c: 0x0,
                        isp_merge_upper_x: F32::from_bits(cmdbuf.isp_merge_upper_x),
                        isp_merge_upper_y: F32::from_bits(cmdbuf.isp_merge_upper_y),
                        unk_18: U64(0x0),
                        utiles_per_mtile_y: tile_info.utiles_per_mtile_y as u16,
                        utiles_per_mtile_x: tile_info.utiles_per_mtile_x as u16,
                        unk_24: 0x0,
                        tile_counts: ((tile_info.tiles_y - 1) << 12) | (tile_info.tiles_x - 1),
                        tib_blocks: blocks_per_utile,
                        isp_bgobjdepth: cmdbuf.isp_bgobjdepth,
                        // TODO: does this flag need to be exposed to userspace?
                        isp_bgobjvals: load_bgobjvals as u32,
                        unk_38: 0x0,
                        unk_3c: 0x1,
                        helper_cfg: cmdbuf.fragment_helper.cfg,
                        __pad: Default::default(),
                    }),
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            r.add(0x1739, 1);
                            r.add(0x10009, utile_config.into());
                            r.add(0x15379, cmdbuf.eot.rsrc_spec.into());
                            r.add(0x15381, cmdbuf.eot.usc.into());
                            r.add(0x15369, cmdbuf.bg.rsrc_spec.into());
                            r.add(0x15371, cmdbuf.bg.usc.into());
                            r.add(0x15131, cmdbuf.isp_merge_upper_x.into());
                            r.add(0x15139, cmdbuf.isp_merge_upper_y.into());
                            r.add(0x100a1, 0);
                            r.add(0x15069, 0);
                            r.add(0x15071, 0); // pointer
                            r.add(0x16058, 0);
                            r.add(0x10019, cmdbuf.ppp_multisamplectl);
                            let isp_mtile_size = (tile_info.utiles_per_mtile_y
                                | (tile_info.utiles_per_mtile_x << 16))
                                .into();
                            r.add(0x100b1, isp_mtile_size); // ISP_MTILE_SIZE
                            r.add(0x16030, isp_mtile_size); // ISP_MTILE_SIZE
                            r.add(
                                0x100d9,
                                (((tile_info.tiles_y - 1) << 12) | (tile_info.tiles_x - 1)).into(),
                            ); // TE_SCREEN
                            r.add(0x16098, inner.scene.tvb_heapmeta_pointer().into());
                            r.add(0x15109, cmdbuf.isp_scissor_base); // ISP_SCISSOR_BASE
                            r.add(0x15101, cmdbuf.isp_dbias_base); // ISP_DBIAS_BASE
                            r.add(0x15021, isp_ctl.into()); // aux_fb_info.unk_1
                            r.add(
                                0x15211,
                                ((cmdbuf.height_px as u64) << 32) | cmdbuf.width_px as u64,
                            ); // aux_fb_info.{width, heigh
                            r.add(0x15049, 0x100000); // s2.aux_fb_info.unk3
                            r.add(0x10051, blocks_per_utile.into()); // s1.unk_2c
                            r.add(0x15321, cmdbuf.isp_zls_pixels.into()); // ISP_ZLS_PIXELS
                            r.add(0x15301, cmdbuf.isp_bgobjdepth.into()); // ISP_BGOBJDEPTH
                            r.add(0x15309, load_bgobjvals); // ISP_BGOBJVALS
                            r.add(0x15311, cmdbuf.isp_oclqry_base); // ISP_OCLQRY_BASE
                            r.add(0x15319, cmdbuf.zls_ctrl); // ISP_ZLSCTL
                            r.add(0x15349, g14_unk); // s2.unk_58_g14_0
                            r.add(0x15351, 0); // s2.unk_58_g14_8
                            r.add(0x15329, cmdbuf.depth.base); // ISP_ZLOAD_BASE
                            r.add(0x15331, cmdbuf.depth.base); // ISP_ZSTORE_BASE
                            r.add(0x15339, cmdbuf.stencil.base); // ISP_STENCIL_LOAD_BASE
                            r.add(0x15341, cmdbuf.stencil.base); // ISP_STENCIL_STORE_BASE
                            r.add(0x15231, 0);
                            r.add(0x15221, 0);
                            r.add(0x15239, 0);
                            r.add(0x15229, 0);
                            r.add(0x15401, cmdbuf.depth.stride as u64); // load
                            r.add(0x15421, cmdbuf.depth.stride as u64); // store
                            r.add(0x15409, cmdbuf.stencil.stride as u64); // load
                            r.add(0x15429, cmdbuf.stencil.stride as u64);
                            r.add(0x153c1, cmdbuf.depth.comp_base); // load
                            r.add(0x15411, cmdbuf.depth.comp_stride as u64); // load
                            r.add(0x153c9, cmdbuf.depth.comp_base); // store
                            r.add(0x15431, cmdbuf.depth.comp_stride as u64); // store
                            r.add(0x153d1, cmdbuf.stencil.comp_base); // load
                            r.add(0x15419, cmdbuf.stencil.comp_stride as u64); // load
                            r.add(0x153d9, cmdbuf.stencil.comp_base); // store
                            r.add(0x15439, cmdbuf.stencil.comp_stride as u64); // store
                            r.add(0x16429, inner.scene.tvb_tilemap_pointer().into());
                            r.add(0x16060, inner.scene.tvb_layermeta_pointer().into());
                            r.add(0x16431, (4 * tile_info.params.rgn_size as u64) << 24); // ISP_RGN?
                            r.add(0x10039, tile_config); // tile_config ISP_CTL?
                            r.add(0x16451, 0x0); // ISP_RENDER_ORIGIN
                            r.add(0x11821, cmdbuf.fragment_helper.binary.into());
                            r.add(0x11829, cmdbuf.fragment_helper.data);
                            r.add(0x11f79, cmdbuf.fragment_helper.cfg.into());
                            r.add(0x15359, 0);
                            r.add(0x10069, self.usc_exec_base); // frag; USC_EXEC_BASE_ISP
                            r.add(0x16020, 0);
                            r.add(0x16461, inner.aux_fb.gpu_pointer().into());
                            r.add(0x16090, inner.aux_fb.gpu_pointer().into());
                            r.add(0x120a1, frg_unk_158);
                            r.add(0x160a8, 0);
                            r.add(0x16068, frg_tilecfg);
                            r.add(0x160b8, 0x0);
                            /*
                            r.add(0x10201, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x10428, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c838, 1);  // ?
                            r.add(0x1ca28, 0x1502960f00); // ??
                            r.add(0x1731, 0x1); // ??
                            */
                        }
                    ),
                    #[ver(G == G15)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |_r| {
                            // Exact G15 command geometry uses the register-list body,
                            // but individual 3D register entries are not yet imported.
                            // Keep the proven/common inputs type-checked without emitting
                            // unverified register programming. Apple also has three optional
                            // G15 registers gated by AGXPerfCtrSampler state; fresh/inactive
                            // sampler state makes that outer gate false, so they are not part
                            // of the base render list and stay absent until G15 perf sampling
                            // has its own independently correct lifecycle.
                            // PM register 0x1ca28 is now fully constructible from the
                            // reconstructed per-slot Parameter Scene Allocations resource. Keep
                            // it non-emitting with the rest of the G15 RegisterArray until the
                            // complete base list crosses the runtime-enablement boundary.
                            // RTM/common formulas independently matched to the existing
                            // kernel-owned geometry producers. Keep these typed here so later
                            // list import cannot silently drift while G15 emission is disabled.
                            let g15_isp_mtile_size: u64 = (tile_info.utiles_per_mtile_y
                                | (tile_info.utiles_per_mtile_x << 16))
                                .into();
                            let g15_te_screen: u64 = tile_info.params.te_screen.into();
                            // Stable hardware-register semantics independently match the
                            // pre-G15 m1n1 map, while Apple G15 emits these sources unchanged.
                            let g15_fb_dimensions: u64 =
                                ((cmdbuf.height_px as u64) << 32) | cmdbuf.width_px as u64;
                            let g15_pixels_per_utile = (cmdbuf.utile_width_px as u32)
                                * (cmdbuf.utile_height_px as u32);
                            let g15_samples_per_utile = g15_pixels_per_utile << samples_log2;
                            let g15_blocks_per_utile =
                                ((cmdbuf.sample_size_B as u32) * g15_samples_per_utile)
                                    .div_ceil(2048);
                            let g15_aux_fb = inner.aux_fb.gpu_pointer();
                            let g15_rgn_stride: u64 = (tile_info.params.rgn_size as u64) << 26;
                            // Exact Apple selectors include private modes 2/3, but Linux's
                            // existing render ABI exposes only the historical normal tiler mode
                            // (`unk1=false`).  That closes the Linux producers at 0x280 /
                            // 0x20_00000000 without guessing private Apple mode semantics.
                            let g15_process_empty_tiles = cmdbuf.flags
                                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES
                                    as u32
                                != 0;
                            let g15_tile_config = g15_fragment_tile_config_linux(
                                cmdbuf.layers as u32,
                                g15_process_empty_tiles,
                            );
                            let g15_tilecfg = g15_fragment_tilecfg_linux(
                                utile_config,
                                cmdbuf.layers as u32,
                                tile_info.params.te_screen,
                            );
                            let g15_pm_scene_reg_1ca28 = g15_pm_scene_slice_gpuva & !0xf;
                            // Apple userspace G15G C0 initializes the raw Render command
                            // with bzero(0x870).  The late +0x648/+0x650/+0x658 fields are
                            // only populated when an MTLRasterizationRateMap implementation
                            // exists; current Asahi UAPI/Mesa expose no such state and Mesa
                            // explicitly disables fragment-shading-rate support.  Raw +0x660
                            // has no normal Render writer after the same zero initialization.
                            // Keep these exact Linux-normal values type-checked while the
                            // complete G15 RegisterArray remains deliberately non-emitting.
                            // Apple raw Render +0x640 is the memoryless-render state:
                            // isMemorylessRender returns framebuffer +0xf75 exactly, and the
                            // only end-pass override (+0x8f3c) is initialized zero with no
                            // producer in the analyzed G15G userspace path. Linux Asahi does
                            // not support memoryless render targets, so the exact current-Linux
                            // value for Fragment 0x1a0b1 (and shared TA 0x1a0a1) is zero.
                            // The raw +0x450/+0x454/+0x458 extension is another
                            // rasterization-rate-map cluster. assignRenderRegisters()
                            // publishes framebuffer +0x1488/+0x1490 only when the same
                            // +0x147c VRS gate is active; current Linux has no such path.
                            // Hence 0x101c1/0x0d469 are zero, while dynamic 0x10791 takes
                            // the normal desc+0x6a1==0 gate branch and equals 0xff0200.
                            // Normal G15 Render leaves raw +0x4bc zero. The
                            // apparent setDepthStencilState() writer is an internal
                            // RenderContext cache at impl+0x9620, not the raw command.
                            // Firmware therefore selects 0x8860; J615 MGPU=1 means
                            // the >=5-MGPU |0x1c contribution is absent.
                            let g15_reg_100b8: u64 = 0x8860;
                            // Raw Render +0x2e8 is the native-resolve auxiliary buffer.
                            // Apple only allocates framebuffer +0x1500 when its native
                            // in-render resolve state (+0x7ca) is active, then publishes it
                            // here conditionally. Honeykrisp resolves in a separate control
                            // stream, so the exact current-Linux value for G15 0x15231 is 0.
                            let g15_native_resolve_reg_15231: u64 = 0;
                            // Raw Render +0x5f0 is covered by the same 0x870-byte bzero.
                            // No AGXRenderCommandRec producer writes it; apparent +0x5f0
                            // stores target framebuffer/program/geometry argument objects.
                            // G15 masks descriptor +0x3a8 (raw +0x5f0) with ~0xff for 0x16058.
                            let g15_raw5f0_reg_16058: u64 = 0;
                            // G15 raw Render +0x388/+0x398 also remain at the command bzero
                            // default. Apparent userspace writers at these literal offsets
                            // target framebuffer/geometry/compute state objects, not the raw
                            // AGXRenderCommandRec. Firmware forwards them as 0x15021/0x15049.
                            let g15_aux_reg_15021: u64 = 0;
                            let g15_aux_reg_15049: u64 = 0;
                            // Raw Render +0x600 survives the 0x870-byte command bzero.
                            // All apparent +0x600 writers in Apple userspace target
                            // RenderContext/ThreadedRenderPass/FramebufferConfig objects;
                            // no AGXRenderCommandRec producer writes this field. Firmware
                            // masks it with 0x1f for both G15 0x120a1 and G15G 0x101e9.
                            let g15_raw600_reg_120a1: u64 = 0;
                            let g15_raw600_reg_101e9: u64 = 0;
                            let g15_vrs_reg_101c1: u64 = 0;
                            let g15_vrs_reg_0d469: u64 = 0;
                            let g15_vrs_reg_10791: u64 = 0xff0200;
                            let g15_memoryless_reg_1a0b1: u64 = 0;
                            let g15_vrs_reg_1a079: u64 = 0;
                            let g15_vrs_reg_1a081: u64 = 0;
                            let g15_vrs_reg_1a0d9: u64 = 0;
                            let g15_vrs_reg_1a0e1: u64 = 0;
                            // Apple copies the sampled-render flag (only 0/1) to raw
                            // Render +0x674. G15 0x1a0f9 masks that dword with
                            // 0xfffffff8, so its exact value is zero in both cases.
                            let g15_sampled_reg_1a0f9: u64 = 0;
                            let _ = (
                                g15_fb_dimensions,
                                g15_blocks_per_utile,
                                g15_aux_fb,
                                frg_unk_158,
                                utile_config,
                                g15_isp_mtile_size,
                                g15_te_screen,
                                g15_rgn_stride,
                                g15_tile_config,
                                g15_tilecfg,
                                g15_pm_record_index,
                                g15_pm_scene_slice_offset,
                                g15_pm_scene_slice_gpuva,
                                g15_pm_scene_reg_1ca28,
                                g15_reg_100b8,
                                g15_native_resolve_reg_15231,
                                g15_raw5f0_reg_16058,
                                g15_aux_reg_15021,
                                g15_aux_reg_15049,
                                g15_raw600_reg_120a1,
                                g15_raw600_reg_101e9,
                                g15_vrs_reg_101c1,
                                g15_vrs_reg_0d469,
                                g15_vrs_reg_10791,
                                g15_memoryless_reg_1a0b1,
                                g15_vrs_reg_1a079,
                                g15_vrs_reg_1a081,
                                g15_vrs_reg_1a0d9,
                                g15_vrs_reg_1a0e1,
                                g15_sampled_reg_1a0f9,
                                g15_pm_record_gpuva,
                                g15_pm_page_metrics_gpuva,
                                load_bgobjvals,
                                inner.scene.tvb_tilemap_pointer(),
                                inner.scene.tvb_heapmeta_pointer(),
                                inner.scene.tvb_layermeta_pointer(),
                            );
                        },
                    ),
                    #[ver(G != G15)]
                    job_params3 <- try_init!(fw::fragment::raw::JobParameters3::ver {
                        isp_dbias_base: fw::fragment::raw::ArrayAddr {
                            ptr: U64(cmdbuf.isp_dbias_base),
                            unk_padding: U64(0),
                        },
                        isp_scissor_base: fw::fragment::raw::ArrayAddr {
                            ptr: U64(cmdbuf.isp_scissor_base),
                            unk_padding: U64(0),
                        },
                        isp_oclqry_base: U64(cmdbuf.isp_oclqry_base),
                        unk_118: U64(0x0),
                        unk_120: Default::default(),
                        unk_partial_bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.partial_bg.rsrc_spec as u64),
                            address: U64(cmdbuf.partial_bg.usc as u64),
                        },
                        unk_258: U64(0),
                        unk_260: U64(0),
                        unk_268: U64(0),
                        unk_270: U64(0),
                        partial_bg: fw::fragment::raw::BackgroundProgram {
                            rsrc_spec: U64(cmdbuf.partial_bg.rsrc_spec as u64),
                            address: U64(cmdbuf.partial_bg.usc as u64),
                        },
                        zls_ctrl: U64(reload_zlsctrl),
                        unk_290: U64(g14_unk),
                        z_load: U64(cmdbuf.depth.base),
                        z_partial_stride: U64(cmdbuf.depth.stride as u64),
                        z_partial_comp_stride: U64(cmdbuf.depth.comp_stride as u64),
                        z_store: U64(cmdbuf.depth.base),
                        z_partial: U64(cmdbuf.depth.base),
                        z_partial_comp: U64(cmdbuf.depth.comp_base),
                        s_load: U64(cmdbuf.stencil.base),
                        s_partial_stride: U64(cmdbuf.stencil.stride as u64),
                        s_partial_comp_stride: U64(cmdbuf.stencil.comp_stride as u64),
                        s_store: U64(cmdbuf.stencil.base),
                        s_partial: U64(cmdbuf.stencil.base),
                        s_partial_comp: U64(cmdbuf.stencil.comp_base),
                        unk_2f8: Default::default(),
                        tib_blocks: blocks_per_utile,
                        unk_30c: 0x0,
                        aux_fb_info,
                        tile_config: U64(tile_config),
                        unk_328_padding: Default::default(),
                        unk_partial_eot: fw::fragment::raw::EotProgram::new(
                            cmdbuf.partial_eot.rsrc_spec,
                            cmdbuf.partial_eot.usc
                        ),
                        partial_eot: fw::fragment::raw::EotProgram::new(
                            cmdbuf.partial_eot.rsrc_spec,
                            cmdbuf.partial_eot.usc
                        ),
                        isp_bgobjdepth: cmdbuf.isp_bgobjdepth,
                        isp_bgobjvals: cmdbuf.isp_bgobjvals,
                        sample_size: cmdbuf.sample_size_B as u32,
                        unk_37c: 0x0,
                        unk_380: U64(0x0),
                        unk_388: U64(0x0),
                        #[ver(V >= V13_0B4)]
                        unk_390_0: U64(0x0),
                        isp_zls_pixels: U64(cmdbuf.isp_zls_pixels as u64),
                    }),
                    #[ver(G == G15)]
                    g15_job_params3: Default::default(),
                    #[ver(G != G15)]
                    unk_758_flag: 0,
                    #[ver(G != G15)]
                    unk_75c_flag: 0,
                    #[ver(G == G15)]
                    g15_zero_a48: 0,
                    #[ver(G == G15)]
                    g15_pad_a4c: 0,
                    #[ver(G != G15)]
                    unk_buf: Default::default(),
                    #[ver(G == G15)]
                    g15_raw_render_2d8_a50: U64(0),
                    #[ver(G == G15)]
                    g15_zero_a58: Default::default(),
                    busy_flag: 0,
                    tvb_overflow_count: 0,
                    unk_878: 0,
                    #[ver(G != G15)]
                    encoder_params <- try_init!(fw::job::raw::EncoderParams {
                        // Maybe set when reloading z/s?
                        unk_8: 0,
                        sync_grow: 0,
                        unk_10: 0x0, // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffffu32,
                        sampler_array: U64(cmdbuf.sampler_heap),
                        sampler_count: cmdbuf.sampler_count as u32,
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    #[ver(G == G15)]
                    g15_encoder_state <- try_init!(fw::fragment::raw::G15EncoderState {
                        zero_b6c: 0,
                        zero_b70: 0,
                        zero_b74: 0,
                        zero_b78: 0,
                        // Apple sources below are exact; Linux has no G15 raw producer yet.
                        raw_render_2dd_b7c: 0,
                        raw_render_1c0_nonzero_b80: 0,
                        raw_render_60c_b84: 0,
                        raw_render_608_610_b88: U64(0),
                        raw_render_614_b90: 0,
                    }),
                    #[ver(G != G15)]
                    process_empty_tiles: (cmdbuf.flags
                        & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES as u32
                        != 0) as u32,
                    #[ver(G != G15)]
                    // TODO: needs to be investigated
                    no_clear_pipeline_textures: 1,
                    #[ver(G != G15)]
                    // TODO: needs to be investigated
                    msaa_zs: 0,
                    #[ver(G != G15)]
                    unk_pointee: 0,
                    #[ver(G == G15)]
                    g15_raw_render_618_b94: 0,
                    #[ver(G == G15)]
                    g15_raw_render_5d8_b98: 0,
                    #[ver(G == G15)]
                    g15_raw_render_5d9_b9c: 0,
                    #[ver(G == G15)]
                    g15_raw_render_4b6_nonzero_ba0: 0,
                    #[ver(V >= V13_3 && G != G15)]
                    unk_v13_3: 0,
                    #[ver(G == G15)]
                    g15_raw_render_4b8_ba4: U32(0),
                    #[ver(G != G15)]
                    meta <- try_init!(fw::job::raw::JobMeta {
                        unk_0: 0,
                        unk_2: 0,
                        no_preemption: no_preemption as u8,
                        stamp: ev_frag.stamp_pointer,
                        fw_stamp: ev_frag.fw_stamp_pointer,
                        stamp_value: ev_frag.value.next(),
                        stamp_slot: ev_frag.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid: uuid_3d,
                        event_seq: ev_frag.event_seq as u32,
                    }),
                    #[ver(G == G15)]
                    meta <- try_init!(fw::job::raw::G15JobMeta {
                        // J615/G15G C0 Apple submission writes raw Render
                        // byte +0x4c0 != 0 to command +0xba8. The accelerator
                        // gate is fixed enabled (+0x1dd8=1), while +0x1dd9
                        // remains zero from zeroed allocation through setup.
                        // Linux has no G15 raw producer yet: fail closed.
                        engine_state: U32(0),
                        stamp: ev_frag.stamp_pointer,
                        fw_stamp: ev_frag.fw_stamp_pointer,
                        stamp_value: ev_frag.value.next(),
                        stamp_slot: ev_frag.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid: uuid_3d,
                        event_seq: ev_frag.event_seq as u32,
                    }),
                    #[ver(G != G15)]
                    unk_after_meta: unk1.into(),
                    #[ver(G == G15)]
                    g15_raw_render_619_eq_2_bd4: U32(0),
                    #[ver(G != G15)]
                    unk_buf_0: U64(0),
                    #[ver(G != G15)]
                    unk_buf_8: U64(0),
                    #[ver(G < G14X && G != G15)]
                    unk_buf_10: U64(1),
                    #[ver(G >= G14X && G != G15)]
                    unk_buf_10: U64(0),
                    #[ver(G == G15)]
                    g15_zero_bd8: U64(0),
                    #[ver(G == G15)]
                    g15_zero_be0: U64(0),
                    #[ver(G == G15)]
                    g15_zero_be8: U64(0),
                    #[ver(G != G15)]
                    command_time: U64(0),
                    #[ver(G == G15)]
                    g15_zero_bf0: U64(0),
                    #[ver(G != G15)]
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), frag.start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), frag.end)),
                    }),
                    #[ver(G != G15)]
                    user_timestamp_pointers: inner.user_timestamps.pointers()?,
                    #[ver(G == G15)]
                    g15_segment_resource_list_fwva_bf8: U64(0),
                    #[ver(G == G15)]
                    g15_block_fence_time0_fwva_c00: U64(0),
                    #[ver(G == G15)]
                    g15_mtl_counter_fw_token_0_c08: U64(0),
                    #[ver(G == G15)]
                    g15_mtl_counter_fw_token_1_c10: U64(0),
                    #[ver(G != G15)]
                    client_sequence: slot_client_seq,
                    #[ver(G != G15)]
                    pad_925: Default::default(),
                    #[ver(G != G15)]
                    unk_928: 0,
                    #[ver(G != G15)]
                    unk_92c: 0,
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_ts: U64(0),
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_92d_8: Default::default(),
                    #[ver(G == G15)]
                    g15_pad_c18: Default::default(),
                    #[ver(G == G15)]
                    g15_uma_page_pool_state_fwva_c1e: U64(0),
                    #[ver(G == G15)]
                    g15_uma_prepared_c26: 0,
                    #[ver(G == G15)]
                    g15_uma_min_pool_size_c27: U64(0),
                    #[ver(G == G15)]
                    g15_uma_ideal_pool_size_c2f: U64(0),
                    #[ver(G == G15)]
                    g15_uma_metrics_fwva_c37: U64(0),
                    #[ver(G == G15)]
                    g15_context_id_generation_c3f: 0,
                    #[ver(G == G15)]
                    g15_sku_timing_c40: Default::default(),
                })
            },
        )?;

        mod_dev_dbg!(self.dev, "[Submission {}] Add Frag\n", id);
        fence.add_command();

        frag_job.add_cb(frag, vm_bind.slot(), move |error| {
            if let Some(err) = error {
                fence.set_error(err.into());
            }

            fence.command_complete();
        })?;

        let fence = job.fence.clone();
        let vtx_job = job.get_vtx()?;

        if scene.rebind() || tvb_grown || tvb_autogrown {
            mod_dev_dbg!(self.dev, "[Submission {}] Create Bind Buffer\n", id);
            let bind_buffer = kalloc.private.new_init(
                {
                    let scene = scene.clone();
                    try_init!(fw::buffer::InitBuffer::ver { scene })
                },
                |inner, _ptr| {
                    let vm_slot = vm_bind.slot();
                    try_init!(fw::buffer::raw::InitBuffer::ver {
                        tag: fw::workqueue::CommandType::InitBuffer,
                        vm_slot,
                        buffer_slot: inner.scene.slot(),
                        unk_c: 0,
                        block_count: buffer.block_count(),
                        buffer: inner.scene.buffer_pointer(),
                        stamp_value: ev_vtx.value.next(),
                    })
                },
            )?;

            mod_dev_dbg!(self.dev, "[Submission {}] Add Bind Buffer\n", id);
            vtx_job.add(bind_buffer, vm_bind.slot())?;
        }

        mod_dev_dbg!(self.dev, "[Submission {}] Create Vertex\n", id);
        let vtx = GpuObject::new_init_prealloc(
            kalloc.gpu_ro.alloc_object()?,
            |ptr: GpuWeakPointer<fw::vertex::RunVertex::ver>| {
                let scene = scene.clone();
                let vm_bind = vm_bind.clone();
                let timestamps = timestamps.clone();
                let private = &mut kalloc.private;
                try_init!(fw::vertex::RunVertex::ver {
                    micro_seq: {
                        let mut builder = microseq::Builder::new();

                        let stats = inner_weak_ptr!(
                            gpu.initdata.runtime_pointers.stats.vtx.weak_pointer(),
                            stats
                        );

                        let start_vtx = builder.add(microseq::StartVertex::ver {
                            header: microseq::op::StartVertex::HEADER,
                            #[ver(G < G14X && G != G15)]
                            tiling_params: Some(inner_weak_ptr!(ptr, tiling_params)),
                            #[ver(G < G14X && G != G15)]
                            job_params1: Some(inner_weak_ptr!(ptr, job_params1)),
                            #[ver(G >= G14X || G == G15)]
                            tiling_params: None,
                            #[ver(G >= G14X || G == G15)]
                            job_params1: None,
                            #[ver(G >= G14X)]
                            registers: inner_weak_ptr!(ptr, registers),
                            buffer: scene.weak_buffer_pointer(),
                            scene: scene.weak_pointer(),
                            stats,
                            work_queue: ev_vtx.info_ptr,
                            vm_slot: vm_bind.slot(),
                            unk_38: 1, // fixed
                            event_generation: self.id as u32,
                            buffer_slot: scene.slot(),
                            unk_44: 0,
                            event_seq: U64(ev_vtx.event_seq),
                            unk_50: 0,
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            #[ver(G != G15)]
                            unk_job_buf: inner_weak_ptr!(ptr, unk_buf_0),
                            #[ver(G == G15)]
                            unk_job_buf: inner_weak_ptr!(ptr, g15_barrier_state_890),
                            unk_64: 0x0, // fixed
                            unk_68: unk1.into(),
                            uuid: uuid_ta,
                            attachments: *vertex_attachments,
                            padding: 0,
                            #[ver(V >= V13_0B4)]
                            counter: U64(count_vtx),
                            #[ver(V >= V13_0B4)]
                            notifier_buf: inner_weak_ptr!(notifier.weak_pointer(), state.unk_buf),
                            #[ver(V < V13_0B4)]
                            unk_178: 0x0, // padding?
                            #[ver(V >= V13_0B4)]
                            unk_178: (!clustering) as u32,
                        })?;

                        #[ver(G != G15)]
                        if vtx_user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(true),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.start_addr),
                                work_queue: ev_vtx.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_ta,
                                unk_30_padding: 0,
                            })?;
                        }

                        #[ver(G < G14X)]
                        builder.add(microseq::WaitForIdle {
                            header: microseq::op::WaitForIdle::new(microseq::Pipe::Vertex),
                        })?;
                        #[ver(G >= G14X)]
                        builder.add(microseq::WaitForIdle2 {
                            header: microseq::op::WaitForIdle2::HEADER,
                        })?;

                        #[ver(G != G15)]
                        if vtx_user_timestamps.any() {
                            builder.add(microseq::Timestamp::ver {
                                header: microseq::op::Timestamp::new(false),
                                command_time: inner_weak_ptr!(ptr, command_time),
                                ts_pointers: inner_weak_ptr!(ptr, timestamp_pointers),
                                update_ts: inner_weak_ptr!(ptr, timestamp_pointers.end_addr),
                                work_queue: ev_vtx.info_ptr,
                                user_ts_pointers: inner_weak_ptr!(ptr, user_timestamp_pointers),
                                #[ver(V >= V13_0B4)]
                                unk_ts: inner_weak_ptr!(ptr, unk_ts),
                                uuid: uuid_ta,
                                unk_30_padding: 0,
                            })?;
                        }

                        let off = builder.offset_to(start_vtx);
                        builder.add(microseq::FinalizeVertex::ver {
                            header: microseq::op::FinalizeVertex::HEADER,
                            scene: scene.weak_pointer(),
                            buffer: scene.weak_buffer_pointer(),
                            stats,
                            work_queue: ev_vtx.info_ptr,
                            vm_slot: vm_bind.slot(),
                            unk_28: 0x0, // fixed
                            unk_pointer: inner_weak_ptr!(ptr, unk_pointee),
                            unk_34: 0x0, // fixed
                            uuid: uuid_ta,
                            fw_stamp: ev_vtx.fw_stamp_pointer,
                            stamp_value: ev_vtx.value.next(),
                            unk_48: U64(0x0), // fixed
                            unk_50: 0x0,      // fixed
                            unk_54: 0x0,      // fixed
                            unk_58: U64(0x0), // fixed
                            unk_60: 0x0,      // fixed
                            unk_64: 0x0,      // fixed
                            unk_68: 0x0,      // fixed
                            #[ver(G >= G14 && V < V13_0B4)]
                            unk_68_g14: U64(0),
                            restart_branch_offset: off,
                            has_attachments: (vertex_attachments.count > 0) as u32,
                            #[ver(V >= V13_0B4)]
                            unk_74: Default::default(), // Ventura
                        })?;

                        builder.add(microseq::RetireStamp {
                            header: microseq::op::RetireStamp::HEADER,
                        })?;
                        builder.build(private)?
                    },
                    notifier,
                    scene,
                    vm_bind,
                    timestamps,
                    user_timestamps: vtx_user_timestamps,
                })
            },
            |inner, _ptr| {
                let vm_slot = vm_bind.slot();
                #[ver(G < G14)]
                let core_masks = gpu.core_masks_packed();

                try_init!(fw::vertex::raw::RunVertex::ver {
                    tag: fw::workqueue::CommandType::RunVertex,
                    #[ver(V >= V13_0B4)]
                    counter: U64(count_vtx),
                    vm_slot,
                    unk_8: 0,
                    notifier: inner.notifier.gpu_pointer(),
                    buffer_slot: inner.scene.slot(),
                    unk_1c: 0,
                    buffer: inner.scene.buffer_pointer(),
                    scene: inner.scene.gpu_pointer(),
                    unk_buffer_buf: inner.scene.kernel_buffer_pointer(),
                    unk_34: 0,
                    #[ver(G < G14X && G != G15)]
                    job_params1 <- try_init!(fw::vertex::raw::JobParameters1::ver {
                        unk_0: U64(if unk1 { 0 } else { 0x200 }), // sometimes 0
                        unk_8: f32!(1e-20),                       // fixed
                        unk_c: f32!(1e-20),                       // fixed
                        tvb_tilemap: inner.scene.tvb_tilemap_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_tilemaps: inner.scene.cluster_tilemaps_pointer(),
                        tpc: inner.scene.tpc_pointer(),
                        tvb_heapmeta: inner.scene.tvb_heapmeta_pointer().or(0x8000_0000_0000_0000),
                        iogpu_unk_54: U64(iogpu_unk54), // fixed
                        iogpu_unk_56: U64(iogpu_unk56), // fixed
                        #[ver(G < G14)]
                        tvb_cluster_meta1: inner
                            .scene
                            .meta_1_pointer()
                            .map(|x| x.or((tile_info.meta1_layer_stride as u64) << 50)),
                        utile_config,
                        unk_4c: 0,
                        ppp_multisamplectl: U64(cmdbuf.ppp_multisamplectl), // fixed
                        tvb_layermeta: inner.scene.tvb_layermeta_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_layermeta: inner.scene.tvb_cluster_layermeta_pointer(),
                        #[ver(G < G14)]
                        core_mask: Array::new([
                            *core_masks.first().unwrap_or(&0),
                            *core_masks.get(1).unwrap_or(&0),
                        ]),
                        preempt_buf1: inner.scene.preempt_buf_1_pointer(),
                        preempt_buf2: inner.scene.preempt_buf_2_pointer(),
                        unk_80: U64(0x1), // fixed
                        preempt_buf3: inner.scene.preempt_buf_3_pointer().or(0x4_0000_0000_0000), // check
                        vdm_ctrl_stream_base: U64(cmdbuf.vdm_ctrl_stream_base),
                        #[ver(G < G14)]
                        tvb_cluster_meta2: inner.scene.meta_2_pointer(),
                        #[ver(G < G14)]
                        tvb_cluster_meta3: inner.scene.meta_3_pointer(),
                        #[ver(G < G14)]
                        tiling_control,
                        #[ver(G < G14)]
                        unk_ac: tiling_control_2 as u32, // fixed
                        unk_b0: Default::default(), // fixed
                        usc_exec_base_ta: U64(self.usc_exec_base),
                        #[ver(G < G14)]
                        tvb_cluster_meta4: inner
                            .scene
                            .meta_4_pointer()
                            .map(|x| x.or(0x3000_0000_0000_0000)),
                        #[ver(G < G14)]
                        unk_f0: U64(vtx_unk_f0),
                        unk_f8: U64(0x8c60),     // fixed
                        helper_program: cmdbuf.vertex_helper.binary,
                        unk_104: 0,
                        helper_arg: U64(cmdbuf.vertex_helper.data),
                        unk_110: Default::default(),      // fixed
                        unk_118: vtx_unk_118 as u32, // fixed
                        __pad: Default::default(),
                    }),
                    #[ver(G < G14X && G != G15)]
                    tiling_params: tile_info.params,
                    #[ver(G >= G14X)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |r| {
                            r.add(0x10141, if unk1 { 0 } else { 0x200 }); // s2.unk_0
                            r.add(0x1c039, inner.scene.tvb_tilemap_pointer().into());
                            r.add(0x1c9c8, inner.scene.tvb_tilemap_pointer().into());

                            let cl_tilemaps_ptr = inner
                                .scene
                                .cluster_tilemaps_pointer()
                                .map_or(0, |a| a.into());
                            r.add(0x1c041, cl_tilemaps_ptr);
                            r.add(0x1c9d0, cl_tilemaps_ptr);
                            r.add(0x1c0a1, inner.scene.tpc_pointer().into()); // TE_TPC_ADDR

                            let tvb_heapmeta_ptr = inner
                                .scene
                                .tvb_heapmeta_pointer()
                                .or(0x8000_0000_0000_0000)
                                .into();
                            r.add(0x1c031, tvb_heapmeta_ptr);
                            r.add(0x1c9c0, tvb_heapmeta_ptr);
                            r.add(0x1c051, iogpu_unk54); // iogpu_unk_54/55
                            r.add(0x1c061, iogpu_unk56); // iogpu_unk_56
                            r.add(0x10149, utile_config.into()); // s2.unk_48 utile_config
                            r.add(0x10139, cmdbuf.ppp_multisamplectl); // PPP_MULTISAMPLECTL
                            r.add(0x10111, inner.scene.preempt_buf_1_pointer().into());
                            r.add(0x1c9b0, inner.scene.preempt_buf_1_pointer().into());
                            r.add(0x10119, inner.scene.preempt_buf_2_pointer().into());
                            r.add(0x1c9b8, inner.scene.preempt_buf_2_pointer().into());
                            r.add(0x1c958, 1); // s2.unk_80
                            r.add(
                                0x1c950,
                                inner
                                    .scene
                                    .preempt_buf_3_pointer()
                                    .or(0x4_0000_0000_0000)
                                    .into(),
                            );
                            r.add(0x1c930, 0); // VCE related addr, lsb to enable
                            r.add(0x1c880, cmdbuf.vdm_ctrl_stream_base); // VDM_CTRL_STREAM_BASE
                            r.add(0x1c898, 0x0); // if lsb set, faults in UL1C0, possibly missing addr.
                            r.add(
                                0x1c948,
                                inner.scene.meta_2_pointer().map_or(0, |a| a.into()),
                            ); // tvb_cluster_meta2
                            r.add(
                                0x1c888,
                                inner.scene.meta_3_pointer().map_or(0, |a| a.into()),
                            ); // tvb_cluster_meta3
                            r.add(0x1c890, tiling_control.into()); // tvb_tiling_control
                            r.add(0x1c918, tiling_control_2);
                            r.add(0x1c079, inner.scene.tvb_layermeta_pointer().into());
                            r.add(0x1c9d8, inner.scene.tvb_layermeta_pointer().into());
                            let cl_layermeta_pointer =
                                inner.scene.tvb_cluster_layermeta_pointer().map_or(0, |a| a.into());
                            r.add(0x1c089, cl_layermeta_pointer);
                            r.add(0x1c9e0, cl_layermeta_pointer);
                            let cl_meta_4_pointer =
                                inner.scene.meta_4_pointer().map_or(0, |a| a.into());
                            r.add(0x16c41, cl_meta_4_pointer); // tvb_cluster_meta4
                            r.add(0x1ca40, cl_meta_4_pointer); // tvb_cluster_meta4
                            r.add(0x1c9a8, vtx_unk_f0); // + meta1_blocks? min_free_tvb_pages?
                            r.add(
                                0x1c920,
                                inner.scene.meta_1_pointer().map_or(0, |a| a.into()),
                            ); // ??? | meta1_blocks?
                            r.add(0x10151, 0);
                            r.add(0x1c199, 0);
                            r.add(0x1c1a1, 0);
                            r.add(0x1c1a9, 0); // 0x10151 bit 1 enables
                            r.add(0x1c1b1, 0);
                            r.add(0x1c1b9, 0);
                            r.add(0x10061, self.usc_exec_base); // USC_EXEC_BASE_TA
                            r.add(0x11801, cmdbuf.vertex_helper.binary.into());
                            r.add(0x11809, cmdbuf.vertex_helper.data);
                            r.add(0x11f71, cmdbuf.vertex_helper.cfg.into());
                            r.add(0x1c0b1, tile_info.params.rgn_size.into()); // TE_PSG
                            r.add(0x1c850, tile_info.params.rgn_size.into());
                            r.add(0x10131, tile_info.params.unk_4.into());
                            r.add(0x10121, tile_info.params.ppp_ctrl.into()); // PPP_CTRL
                            r.add(
                                0x10129,
                                tile_info.params.x_max as u64
                                    | ((tile_info.params.y_max as u64) << 16),
                            ); // PPP_SCREEN
                            r.add(0x101b9, tile_info.params.te_screen.into()); // TE_SCREEN
                            r.add(0x1c069, tile_info.params.te_mtile1.into()); // TE_MTILE1
                            r.add(0x1c071, tile_info.params.te_mtile2.into()); // TE_MTILE2
                            r.add(0x1c081, tile_info.params.tiles_per_mtile.into()); // TE_MTILE
                            r.add(0x1c0a9, tile_info.params.tpc_stride.into()); // TE_TPC
                            r.add(0x10171, tile_info.params.unk_24.into());
                            r.add(0x10169, tile_info.params.unk_28.into()); // TA_RENDER_TARGET_MAX
                            r.add(0x12099, vtx_unk_118);
                            r.add(0x1c9e8, (tile_info.params.unk_28 & 0x4fff).into());
                            /*
                            r.add(0x10209, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c9f0, 0x100); // Some kind of counter?? Does this matter?
                            r.add(0x1c830, 1); // ?
                            r.add(0x1ca30, 0x1502960e60); // ?
                            r.add(0x16c39, 0x1502960e60); // ?
                            r.add(0x1c910, 0xa0000b011d); // ?
                            r.add(0x1c8e0, 0xff); // cluster mask
                            r.add(0x1c8e8, 0); // ?
                            */
                        }
                    ),
                    #[ver(G == G15)]
                    registers: fw::job::raw::RegisterArray::new(
                        inner_weak_ptr!(_ptr, registers.registers),
                        |_r| {
                            // Keep the existing helper inputs type-checked for the
                            // later G15 register reconstruction without emitting
                            // unproven register entries into this compile-only shell.
                            // Apple TA descriptor +0xe58 is RTM +0x2a8, the same TVB
                            // heap-metadata address used by Fragment 0x16098. Its
                            // 0x1c031/0x1c9c0 tag is controlled by accelerator +0x650
                            // bit 21. Base configureDevice() clears that bit and G15/G15G
                            // never set it, so exact J615 keeps the high-bit tag set.
                            let g15_ta_tiler_mode_10141 = G15_LINUX_TILER_MODE_10141;
                            let g15_ta_tilemap = inner.scene.tvb_tilemap_pointer();
                            let g15_ta_layermeta = inner.scene.tvb_layermeta_pointer();
                            let g15_ta_rgn_size: u64 = tile_info.params.rgn_size.into();
                            // RTM +0xf0 independently reconstructs the same utile/sample
                            // encoding as Linux `utile_config`. RTM +0xf8 is packed from
                            // the same AGXSampleOffsetRec as Fragment RTM +0xb0, making
                            // TA 0x10139 the same PPP_MULTISAMPLECTL value.
                            let g15_ta_utile_config: u64 = utile_config.into();
                            let g15_ta_ppp_multisamplectl = cmdbuf.ppp_multisamplectl;
                            // Apple G15 synthesizes 0x10121 (PPP_CTRL) from private raw
                            // Render fields plus accelerator feature state.  For normal J615
                            // that yields 0x202, exactly matching Mesa's named W-clamp +
                            // fixed-point-format value.  Linux UAPI already carries the final
                            // hardware PPP_CTRL value, as it does for earlier generations, so
                            // the Linux-side producer is the existing raw register value.
                            // Apple-only exceptional 0x600e/0x800 modes remain documented
                            // separately and do not need to be invented by the kernel.
                            let g15_ta_ppp_ctrl: u64 = cmdbuf.ppp_ctrl.into();
                            // Exact G15 RTM geometry maps back onto Linux TilingParameters.
                            let g15_ta_ppp_screen: u64 = tile_info.params.x_max as u64
                                | ((tile_info.params.y_max as u64) << 16);
                            let g15_ta_te_screen: u64 = tile_info.params.te_screen.into();
                            let g15_ta_te_mtile1: u64 = tile_info.params.te_mtile1.into();
                            let g15_ta_te_mtile2: u64 = tile_info.params.te_mtile2.into();
                            let g15_ta_tiles_per_mtile: u64 =
                                tile_info.params.tiles_per_mtile.into();
                            let g15_ta_tpc_stride: u64 = tile_info.params.tpc_stride.into();
                            // Apple G15 desc +0xe18 is the first GTP-only RTM subregion.
                            // Its size is 0x80 * utiles_per_mtile * layers * MGPUs,
                            // algebraically matching Linux TPC storage, and the established
                            // G14X RegisterArray maps the same 0x1c0a1 register to TE_TPC_ADDR.
                            // Keep this producer type-checked only until the G15 array is
                            // enabled as a whole.
                            let g15_ta_tpc_pointer = inner.scene.tpc_pointer();
                            let g15_ta_geom_const_88: u64 = 0x88;
                            let g15_ta_geom_const_100: u64 = 0x100;
                            let g15_ta_process_empty_tiles = cmdbuf.flags
                                & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES
                                    as u32
                                != 0;
                            let g15_ta_render_target_max = g15_ta_render_target_max(
                                cmdbuf.layers as u32,
                                g15_ta_process_empty_tiles,
                            );
                            let g15_ta_render_target_max_masked =
                                g15_ta_render_target_max & 0x47ff;
                            let g15_ta_heapmeta_tagged = inner
                                .scene
                                .tvb_heapmeta_pointer()
                                .or(0x8000_0000_0000_0000);
                            // Exact Apple PM tail of the normal TA list:
                            // 0x1ca30 = record+0x28 & ~0xf;
                            // encoded 32-bit 0x16c39 carries the same source value;
                            // 0x1c910 is synthesized from record+0x00 (the selected
                            // four-byte PMPageMetricsBuffer slot in shared range 7).
                            let g15_ta_pm_scene = g15_pm_scene_slice_gpuva & !0xf;
                            let g15_ta_pm_metrics_1c910 =
                                buffer::g15_pm_page_metrics_reg_1c910(g15_pm_page_metrics_gpuva);
                            let _ = (
                                g15_ta_tiler_mode_10141,
                                // RTM-backed G15 TA producer pairs:
                                // 0x1c039/0x1c9c8 = tilemap;
                                // 0x1c079/0x1c9d8 = layer metadata;
                                // 0x1c0b1/0x1c850 = region-size dword;
                                // 0x1c031/0x1c9c0 = tagged heap metadata.
                                g15_ta_tilemap,
                                g15_ta_layermeta,
                                g15_ta_rgn_size,
                                g15_ta_utile_config,
                                g15_ta_ppp_multisamplectl,
                                g15_ta_ppp_ctrl,
                                g15_ta_ppp_screen,
                                g15_ta_te_screen,
                                g15_ta_te_mtile1,
                                g15_ta_te_mtile2,
                                g15_ta_tiles_per_mtile,
                                g15_ta_tpc_stride,
                                g15_ta_tpc_pointer,
                                g15_ta_geom_const_88,
                                g15_ta_geom_const_100,
                                g15_ta_render_target_max,
                                g15_ta_render_target_max_masked,
                                g15_ta_heapmeta_tagged,
                                g15_ta_pm_scene,
                                g15_ta_pm_metrics_1c910,
                                g15_pm_record_index,
                                g15_pm_scene_slice_offset,
                                g15_pm_scene_slice_gpuva,
                                g15_pm_page_metrics_gpuva,
                                iogpu_unk54,
                                iogpu_unk56,
                                vtx_unk_118,
                                inner.scene.preempt_buf_1_pointer(),
                                inner.scene.preempt_buf_2_pointer(),
                                inner.scene.preempt_buf_3_pointer(),
                            );
                        },
                    ),
                    #[ver(G == G15)]
                    g15_pad_750: Default::default(),
                    tpc: inner.scene.tpc_pointer(),
                    tpc_size: U64(tile_info.tpc_size as u64),
                    microsequence: inner.micro_seq.gpu_pointer(),
                    microsequence_size: inner.micro_seq.len() as u32,
                    fragment_stamp_slot: ev_frag.slot,
                    fragment_stamp_value: ev_frag.value.next(),
                    unk_pointee: 0,
                    #[ver(G != G15)]
                    unk_pad: 0,
                    #[ver(G != G15)]
                    job_params2 <- try_init!(fw::vertex::raw::JobParameters2 {
                        unk_480: Default::default(), // fixed
                        unk_498: U64(0x0),           // fixed
                        unk_4a0: 0x0,                // fixed
                        preempt_buf1: inner.scene.preempt_buf_1_pointer(),
                        unk_4ac: 0x0,      // fixed
                        unk_4b0: U64(0x0), // fixed
                        unk_4b8: 0x0,      // fixed
                        unk_4bc: U64(0x0), // fixed
                        unk_4c4_padding: Default::default(),
                        unk_50c: 0x0,      // fixed
                        unk_510: U64(0x0), // fixed
                        unk_518: U64(0x0), // fixed
                        unk_520: U64(0x0), // fixed
                    }),
                    #[ver(G != G15)]
                    encoder_params <- try_init!(fw::job::raw::EncoderParams {
                        unk_8: 0x0,     // fixed
                        sync_grow: 0x0, // fixed
                        unk_10: 0x0,    // fixed
                        encoder_id: 0,
                        unk_18: 0x0, // fixed
                        unk_mask: 0xffffffffu32,
                        sampler_array: U64(cmdbuf.sampler_heap),
                        sampler_count: cmdbuf.sampler_count as u32,
                        sampler_max: (cmdbuf.sampler_count as u32) + 1,
                    }),
                    #[ver(G != G15)]
                    unk_55c: 0,
                    #[ver(G != G15)]
                    unk_560: 0,
                    #[ver(G != G15)]
                    sync_grow: 0,
                    #[ver(G != G15)]
                    unk_568: 0,
                    #[ver(G != G15)]
                    uses_scratch: (cmdbuf.flags
                        & uapi::drm_asahi_render_flags_DRM_ASAHI_RENDER_VERTEX_SCRATCH as u32
                        != 0) as u32,
                    #[ver(G == G15)]
                    g15_raw_render_10_788: U64(0),
                    #[ver(G == G15)]
                    g15_raw_render_18_790: U64(0),
                    #[ver(G == G15)]
                    g15_raw_render_20_798: U64(0),
                    #[ver(G == G15)]
                    g15_raw_render_28_7a0: U64(0),
                    #[ver(G == G15)]
                    g15_raw_render_60_7a8: U64(0),
                    #[ver(G == G15)]
                    g15_zero_7b0: U64(0),
                    #[ver(G == G15)]
                    g15_zero_7b8: U64(0),
                    #[ver(G == G15)]
                    g15_zero_7c0: U64(0),
                    #[ver(G == G15)]
                    g15_pad_7c8: Default::default(),
                    #[ver(G == G15)]
                    g15_zero_810: Default::default(),
                    #[ver(G == G15)]
                    g15_pad_830: Default::default(),
                    #[ver(G == G15)]
                    g15_raw_render_60c_83c: U32(0),
                    #[ver(G == G15)]
                    g15_raw_render_608_610_lo_840: U64(0),
                    #[ver(G == G15)]
                    g15_raw_render_614_848: U32(0),
                    #[ver(G == G15)]
                    g15_raw_render_1bc_nonzero_84c: U32(0),
                    #[ver(G == G15)]
                    g15_raw_render_1bf_850: U32(0),
                    #[ver(G == G15)]
                    g15_segment_flag_198_854: U32(0),
                    #[ver(G == G15)]
                    g15_raw_render_1c1_858: U32(0),
                    #[ver(G == G15)]
                    g15_raw_render_1c2_85c: U32(0),
                    #[ver(G != G15)]
                    meta <- try_init!(fw::job::raw::JobMeta {
                        unk_0: 0,
                        unk_2: 0,
                        no_preemption: no_preemption as u8,
                        stamp: ev_vtx.stamp_pointer,
                        fw_stamp: ev_vtx.fw_stamp_pointer,
                        stamp_value: ev_vtx.value.next(),
                        stamp_slot: ev_vtx.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid: uuid_ta,
                        event_seq: ev_vtx.event_seq as u32,
                    }),
                    #[ver(G == G15)]
                    meta <- try_init!(fw::job::raw::G15JobMeta {
                        // J615/G15G C0 Apple TA submission leaves +0x860 zero:
                        // accelerator packed feature bit 35 is provably zero.
                        // Descriptor +0xe08 is (raw Render[0x1bc] != 0), and
                        // ChinookV9 isForceGTPDiscardEnabled() is false, but
                        // the bit-35 gate prevents either path from enabling it.
                        engine_state: U32(0),
                        stamp: ev_vtx.stamp_pointer,
                        fw_stamp: ev_vtx.fw_stamp_pointer,
                        stamp_value: ev_vtx.value.next(),
                        stamp_slot: ev_vtx.slot,
                        evctl_index: 0, // fixed
                        flush_stamps: flush_stamps as u32,
                        uuid: uuid_ta,
                        event_seq: ev_vtx.event_seq as u32,
                    }),
                    #[ver(G != G15)]
                    unk_after_meta: unk1.into(),
                    #[ver(G == G15)]
                    g15_raw_render_619_eq_2_88c: U32(0),
                    #[ver(G != G15)]
                    unk_buf_0: U64(0),
                    #[ver(G != G15)]
                    unk_buf_8: U64(0),
                    #[ver(G != G15)]
                    unk_buf_10: U64(0),
                    #[ver(G != G15)]
                    command_time: U64(0),
                    #[ver(G != G15)]
                    timestamp_pointers <- try_init!(fw::job::raw::TimestampPointers {
                        start_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), vtx.start)),
                        end_addr: Some(inner_ptr!(inner.timestamps.gpu_pointer(), vtx.end)),
                    }),
                    #[ver(G != G15)]
                    user_timestamp_pointers: inner.user_timestamps.pointers()?,
                    #[ver(G != G15)]
                    client_sequence: slot_client_seq,
                    #[ver(G != G15)]
                    pad_5d5: Default::default(),
                    #[ver(G != G15)]
                    unk_5d8: 0,
                    #[ver(G != G15)]
                    unk_5dc: 0,
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_ts: U64(0),
                    #[ver(V >= V13_0B4 && G != G15)]
                    unk_5dd_8: Default::default(),
                    #[ver(G == G15)]
                    g15_barrier_state_890: U64(0),
                    #[ver(G == G15)]
                    g15_pad_898: Default::default(),
                    #[ver(G == G15)]
                    g15_zero_8a8: U64(0),
                    #[ver(G == G15)]
                    g15_segment_resource_list_fwva_8b0: U64(0),
                    #[ver(G == G15)]
                    g15_block_fence_time0_fwva_8b8: U64(0),
                    #[ver(G == G15)]
                    g15_mtl_counter_fw_token_0_8c0: U64(0),
                    #[ver(G == G15)]
                    g15_mtl_counter_fw_token_1_8c8: U64(0),
                    #[ver(G == G15)]
                    g15_pad_8d0: Default::default(),
                    #[ver(G == G15)]
                    g15_uma_page_pool_state_fwva_8d6: U64(0),
                    #[ver(G == G15)]
                    g15_uma_prepared_8de: 0,
                    #[ver(G == G15)]
                    g15_uma_min_pool_size_8df: U64(0),
                    #[ver(G == G15)]
                    g15_uma_ideal_pool_size_8e7: U64(0),
                    #[ver(G == G15)]
                    g15_uma_metrics_fwva_8ef: U64(0),
                    #[ver(G == G15)]
                    g15_context_id_generation_8f7: 0,
                    #[ver(G == G15)]
                    g15_sku_timing_8f8: Default::default(),
                    #[ver(G == G15)]
                    g15_tail_918: Default::default(),
                })
            },
        )?;

        core::mem::drop(alloc);

        mod_dev_dbg!(self.dev, "[Submission {}] Add Vertex\n", id);
        fence.add_command();
        vtx_job.add_cb(vtx, vm_bind.slot(), move |error| {
            if let Some(err) = error {
                fence.set_error(err.into())
            }

            fence.command_complete();
        })?;

        mod_dev_dbg!(self.dev, "[Submission {}] Increment counters\n", id);

        // TODO: handle rollbacks, move to job submit?
        buffer.increment();

        job.get_vtx()?.next_seq();
        job.get_frag()?.next_seq();

        Ok(())
    }
}
