pub mod dataset;
pub mod distribution;
pub mod apis;
pub mod requester;
pub mod token_sampler;

use core::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::yield_now;

use tracing::{instrument, Level};

pub const TTFT: f32 = 5.;   // 5s
pub const TPOT: f32 = 0.06; // 60ms

pub fn timeout_secs_upon_slo(output_length: u64) -> u64 {
    15.max((TTFT + TPOT * output_length as f32) as u64)
}

/// Light weighted spinlock, for extremely short critical section
/// do not abuse it
pub struct SpinLock {
    flag: AtomicBool, // false: unlocked, true: locked
}

unsafe impl Send for SpinLock {}
unsafe impl Sync for SpinLock {}

#[allow(unused)]
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

unsafe impl Send for SpinRwLock {}
unsafe impl Sync for SpinRwLock {}

pub struct SpinRwLock {
    state: AtomicUsize,
}

const USIZE_BITS: u32 = (core::mem::size_of::<usize>() * 8) as u32;
const WRITER_BIT: usize = 1usize << (USIZE_BITS - 1);
const WAITER_BIT: usize = 1usize << (USIZE_BITS - 2);
const READER_MASK: usize = !(WRITER_BIT | WAITER_BIT);

impl SpinRwLock {
    pub const fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
        }
    }

    /// Get read lock, while writer is priorized
    #[instrument(skip_all, level = Level::DEBUG, target = "spin_rwlck::read")]
    pub fn read_lock(&self) {
        let mut spins = 0u32;
        loop {
            let s = self.state.load(Ordering::Relaxed);
            if s & (WRITER_BIT | WAITER_BIT) != 0 {
                // can't acquire lock due to writer
                spins += 1;
                backoff(spins);
                continue;
            }
            if self
                .state
                .compare_exchange_weak(
                    s,
                    s + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // acquire read lock, add reader counter
                return;
            }
            // can't acquire lock due to writer
            spins += 1;
            backoff(spins);
        }
    }

    pub fn read_unlock(&self) {
        // sub reader counter
        let prev = self.state.fetch_sub(1, Ordering::Release);
        debug_assert!(prev & READER_MASK >= 1);
    }

    /// Get write lock, while writer is priorized
    #[instrument(skip_all, level = Level::DEBUG, target = "spin_rwlck::write")]
    pub fn write_lock(&self) {
        let mut spins = 0u32;
        // mark self as waiter
        loop {
            let s = self.state.load(Ordering::Relaxed);
            if s & WAITER_BIT == 0 {
                if self
                    .state
                    .compare_exchange_weak(
                        s,
                        s | WAITER_BIT,
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // self is the waiter now
                    break;
                }
            } else {
                // other writer is the waiter, wait for write lock
                spins += 1;
                backoff(spins);
            }
        }

        let mut spins = 0u32;
        loop {
            let s = self.state.load(Ordering::Relaxed);
            if s & READER_MASK == 0 && s & WRITER_BIT == 0 {
                // precond: self is the write lock waiter
                // no readers hold lock, no writer holds lock
                if self
                    .state
                    .compare_exchange(
                        WAITER_BIT,
                        WRITER_BIT,
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // acquire writer lock, set writer bit
                    return;
                }
            }
            spins += 1;
            backoff(spins);
        }
    }

    pub fn write_unlock(&self) {
        // clear writer bit
        let prev = self.state.swap(0, Ordering::Release);
        debug_assert!(prev & WRITER_BIT != 0);
    }
}
