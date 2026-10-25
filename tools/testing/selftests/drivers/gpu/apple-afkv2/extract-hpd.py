#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only OR MIT
"""Exercise the production DPTX HPD request with mocked service-call replies."""
import pathlib
import re
import sys


def function(source, name):
    match = re.search(r"^(?:static )?int " + name + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError("production definition of " + name + " not found")
    end, depth = match.end(), 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    # Older trees keep the call body static; the unit declares it below.
    return re.sub(r"^static ", "", source[match.start():end])


afk = pathlib.Path(sys.argv[1]).read_text()
dptx = pathlib.Path(sys.argv[2]).read_text()
command = re.search(r"^struct dcpdptx_hotplug_cmd \{.*?\} __attribute__\(\(packed\)\);",
                    dptx, re.M | re.S)
if not command:
    raise RuntimeError("production HPD request layout not found")
prefix = r'''
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "afkv2-codec.h"
typedef u32 __le32;
#define GFP_KERNEL 0
#define kzalloc(n,f) calloc(1,n)
#define kfree(p) free(p)
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define cpu_to_le16(x) (x)
#define cpu_to_le32(x) (x)
#define le16_to_cpu(x) (x)
#define le32_to_cpu(x) (x)
#define EPIC_SERVICE_CALL_MAGIC 0x69706378
#define EPIC_SUBTYPE_STD_SERVICE 0xc0
#define dev_warn(...) do {} while (0)
#define dev_dbg(...) do {} while (0)
struct epic_service_call {
 u8 pad0[2]; u16 group; u32 command, data_len, magic; u8 pad1[48];
};
struct dcp { void *dev; };
struct ops { const char *name; };
struct ep { struct dcp *dcp; int endpoint; bool inline_v2; };
struct apple_epic_service { struct ep *ep; struct ops *ops; u32 channel; };
int afk_service_call_timeout_len(struct apple_epic_service *service, u16 group,
 u32 command, const void *data, size_t data_len, size_t data_pad, void *output,
 size_t output_len, size_t output_pad, unsigned int timeout_ms, size_t *received);
int afk_service_call_timeout(struct apple_epic_service *service, u16 group,
 u32 command, const void *data, size_t data_len, size_t data_pad, void *output,
 size_t output_len, size_t output_pad, unsigned int timeout_ms);
int dptxport_set_hpd_timeout(struct apple_epic_service *service, bool hpd,
 unsigned int timeout_ms);
/*
 * A matched, successful reply in the request's buffer: a 64-byte header, then
 * the body the firmware declares, of which the echo is the word at 16.  An
 * AFK-v2 transport reports how much of the buffer arrived; a legacy one fills
 * it.  Without a length query AFK-v2 refuses anything short of the buffer.
 */
static bool wire_v2;
static size_t declared, arrived;
static u32 echo_flip;
static int afk_send_command_timeout_len(struct apple_epic_service *s, u8 type,
 const void *payload, size_t payload_len, void *output, size_t output_len,
 u32 *retcode, unsigned int timeout_ms, size_t *received)
{
 struct epic_service_call *call = output;
 u8 *body = (u8 *)output + sizeof(*call);
 u32 echo;

 (void)s; (void)payload; (void)timeout_ms;
 assert(type == EPIC_SUBTYPE_STD_SERVICE && payload_len == output_len);
 assert(output_len == sizeof(*call) + 32 && call->data_len == 32);
 assert(call->group == 8 && call->command == 8);
 if (wire_v2 && !received && arrived < output_len - sizeof(*call))
  return -EMSGSIZE;
 memcpy(&echo, body + 16, sizeof(echo));
 echo ^= echo_flip;
 memcpy(body + 16, &echo, sizeof(echo));
 call->data_len = declared;
 *retcode = 0;
 if (received)
  *received = sizeof(*call) + (wire_v2 ? arrived : output_len - sizeof(*call));
 return 0;
}
'''
suffix = r'''
static int failures;
static void expect(bool ok, const char *what, bool hpd, int ret)
{
 printf("%s %s: %s hpd=%d declared=%zu arrived=%zu ret=%d\n", ok ? "ok" : "FAIL",
        wire_v2 ? "afk-v2" : "legacy", what, hpd, declared, arrived, ret);
 failures += !ok;
}
int main(void)
{
 static const size_t lengths[] = { 0, 4, 8, 12, 16, 20, 32, 36, 65535 };
 struct dcp dcp = { 0 };
 struct ops ops = { "AppleDCPDPTXRemotePort" };
 struct ep ep = { .dcp = &dcp };
 struct apple_epic_service service = { .ep = &ep, .ops = &ops, .channel = 1 };

 for (int v2 = 0; v2 < 2; v2++) {
  wire_v2 = v2;
  for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
   for (int hpd = 0; hpd < 2; hpd++) {
    int ret;

    declared = lengths[i]; arrived = 32; echo_flip = 0;
    ret = dptxport_set_hpd_timeout(&service, hpd, 1000);
    if (declared < sizeof(struct dcpdptx_hotplug_cmd))
     expect(ret < 0, "short reply refused", hpd, ret);
    else if (declared > 32)
     expect(ret < 0, "reply longer than its buffer refused", hpd, ret);
    else
     expect(ret == 0, "complete reply accepted", hpd, ret);
   }
  }
  for (int hpd = 0; hpd < 2; hpd++) {
   int ret;

   declared = 20; arrived = 32; echo_flip = 1;
   ret = dptxport_set_hpd_timeout(&service, hpd, 1000);
   expect(ret == -EINVAL, "wrong echo refused", hpd, ret);
   if (!wire_v2)
    continue;
   declared = 20; arrived = 16; echo_flip = 0;
   ret = dptxport_set_hpd_timeout(&service, hpd, 1000);
   expect(ret < 0, "reply cut short in transit refused", hpd, ret);
  }
 }
 printf("%d HPD reply checks failed\n", failures);
 if (failures)
  return 1;
 puts("PASS: extracted production DPTX HPD request: short, oversized, wrong-echo and complete replies, legacy and AFK-v2");
 return 0;
}
'''
unit = (prefix + command.group(0) + "\n" + function(afk, "afk_service_call_timeout_len") + "\n" +
        function(afk, "afk_service_call_timeout") + "\n" +
        function(dptx, "dptxport_set_hpd_timeout") + suffix)
pathlib.Path(sys.argv[3]).write_text(unit)
