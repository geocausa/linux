# SPDX-License-Identifier: GPL-2.0-only OR MIT
# Copyright 2026 Ryan Murray
"""RTKit event dispatch and ACK policies with public synthetic messages."""
from common import block

def oslog(root):
    source = (root / 'drivers/soc/apple/rtkit.c').read_text()
    constants = '\n'.join(line for line in source.splitlines()
                          if line.startswith('#define APPLE_RTKIT_OSLOG_'))
    return r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef uint64_t u64;
typedef uint8_t u8;
#define GENMASK_ULL(h,l) ((~0ULL << (l)) & (~0ULL >> (63-(h))))
#define FIELD_GET(mask,value) (((value) & (mask)) >> __builtin_ctzll(mask))
#define APPLE_RTKIT_EP_OSLOG 8
struct buffer { unsigned sentinel; unsigned size; };
struct apple_rtkit { void *dev; struct buffer oslog_buffer; bool oslog_inherited, sync_syslog_ack; };
static unsigned warnings, notifications, requests, acks, errors;
static u64 observed;
static bool neo, m5;
#define of_machine_is_compatible(s) ((!strcmp(s,"apple,t8140") && neo) || (!strcmp(s,"apple,t8142") && m5))
#define dev_warn(dev,fmt,msg) do { (void)(dev); (void)(fmt); warnings++; observed=(msg); } while (0)
#define dev_dbg(dev,fmt,msg) do { (void)(dev); (void)(fmt); notifications++; observed=(msg); } while (0)
#define dev_err(dev,fmt,msg) do { (void)(dev); (void)(fmt); (void)(msg); errors++; } while (0)
static int __attribute__((unused)) apple_rtkit_send_message(struct apple_rtkit *rtk, u8 ep, u64 msg, void *completion, bool atomic)
{ (void)rtk; assert(ep==8 && !completion && !atomic); acks++; observed=msg; return 0; }
static void apple_rtkit_common_rx_get_buffer(struct apple_rtkit *rtk, struct buffer *b, u8 ep, u64 msg)
{ assert(b==&rtk->oslog_buffer && ep==8); requests++; observed=msg; }
''' + constants + '\n' + block(source, 'static void apple_rtkit_oslog_rx(') + r'''
static void reset(void) { warnings=notifications=requests=acks=errors=0; observed=0; }
int main(void)
{
    struct apple_rtkit rtk={.oslog_buffer={.sentinel=0x12345678}};
    neo=true;
    for (u8 type=2; type<=5; type++) {
        for (unsigned edge=0; edge<2; edge++) {
            u64 msg=((u64)type<<56) | (edge ? ((1ULL<<56)-1) : 0);
            reset(); apple_rtkit_oslog_rx(&rtk,msg);
            assert(!warnings && notifications==1 && !requests && !acks && !errors);
            assert(observed==msg);
            assert(rtk.oslog_buffer.sentinel==0x12345678);
        }
    }
    puts("PASS recognized opaque types preserve both payload bounds without buffer/ACK operations");
    const u8 unknown[]={0,6,0xff};
    for (unsigned i=0;i<sizeof(unknown);i++) {
        u64 msg=((u64)unknown[i]<<56) | 42;
        reset(); apple_rtkit_oslog_rx(&rtk,msg);
        assert(warnings==1 && !notifications && !requests && observed==msg);
    }
    puts("PASS unknown types remain warnings");
    reset(); apple_rtkit_oslog_rx(&rtk,1ULL<<56);
    assert(requests==1 && !warnings && !notifications && observed==(1ULL<<56));
    puts("PASS buffer requests retain existing dispatch");
#ifdef APPLE_RTKIT_OSLOG_LOG
    /* Other machines acknowledge only admitted channel-zero counters. */
    neo=m5=false;
    reset(); apple_rtkit_oslog_rx(&rtk,2ULL<<56);
    assert(warnings==1 && !acks && !notifications);
    for (unsigned inherited=0; inherited<2; inherited++) {
        rtk.oslog_buffer.size=inherited ? 0 : 4096;
        rtk.oslog_inherited=inherited;
        reset(); apple_rtkit_oslog_rx(&rtk,(2ULL<<56)|UINT32_MAX);
        assert(acks==1 && !warnings && observed==((2ULL<<56)|UINT32_MAX));
        reset(); apple_rtkit_oslog_rx(&rtk,(2ULL<<56)|(1ULL<<32));
        assert(warnings==1 && !acks);
    }
    /* Each tested platform policy overrides channel and buffer admission. */
    rtk.oslog_inherited=false;
    for (unsigned policy=0; policy<3; policy++) {
        neo=policy==0; m5=policy==1; rtk.sync_syslog_ack=policy==2;
        reset(); apple_rtkit_oslog_rx(&rtk,(2ULL<<56)|(1ULL<<32));
        assert(notifications==1 && !warnings && !acks);
    }
    puts("PASS channel-zero ACK admission and Neo/M5/synchronous no-ACK policies");
#else
    (void)neo; (void)m5;
#endif
    return 0;
}
'''
