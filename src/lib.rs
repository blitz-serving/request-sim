pub mod dataset;
pub mod distribution;
pub mod protocols;
pub mod requester;
pub mod token_sampler;

use core::hint::spin_loop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::yield_now;

/// 轻量级自旋锁（适合临界区极短、低竞争/不可阻塞场景；
/// 如果存在阻塞或临界区较长，优先考虑 std::sync::Mutex）
pub struct SpinLock {
    flag: AtomicBool, // false: unlocked, true: locked
}

unsafe impl Send for SpinLock {}
unsafe impl Sync for SpinLock {}

impl SpinLock {
    pub const fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
        }
    }

    /// 阻塞式获取锁（自旋）
    pub fn lock(&self) {
        // test-and-test-and-set + 退避
        let mut spins = 0u32;
        loop {
            // 快路径：先“读”观察是否可能解锁（避免频繁写入导致的总线抖动）
            while self.flag.load(Ordering::Relaxed) {
                // 小退避：短自旋
                spins = backoff(spins);
            }

            // 真正尝试：CAS 抢锁
            match self.flag.compare_exchange(
                false,
                true,
                Ordering::Acquire, // 成功获取，建立 Acquire 栅栏
                Ordering::Relaxed, // 失败分支可用 Relaxed
            ) {
                Ok(_) => break,
                Err(_) => {
                    // 失败则继续退避
                    spins = backoff(spins);
                }
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        // 释放锁：Release 保证写入 data 对后继获取者可见
        self.flag.store(false, Ordering::Release);
    }
}

/// 指数回退：前几次 busy-spin，之后主动让出时间片
#[inline]
fn backoff(spins: u32) -> u32 {
    // 前 64 次：CPU hint
    if spins < 64 {
        spin_loop();
        spins + 1
    } else {
        // 偶尔让出 CPU，避免长期霸占
        if spins & 0xF == 0 {
            // 每 16 次让出一次
            yield_now();
        } else {
            spin_loop();
        }
        spins.saturating_add(1)
    }
}
