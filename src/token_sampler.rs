use core::panic;
use crossbeam::channel;
use rand::Rng;
use serde_json::Value;
use tracing::{instrument, span, Level};
use std::{
    collections::HashMap,
    fs,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tokenizers::Tokenizer;

/// TokenSampler: 带异步采样与缓存机制的随机文本块生成器
pub struct TokenSampler {
    tokenizer: Tokenizer,
    vocab_size: u32,
    splitter: Vec<String>,
    block_size: u32,
    receiver: Arc<Mutex<channel::Receiver<String>>>,
    ragged_block_cache: Arc<Mutex<HashMap<usize, channel::Receiver<String>>>>,
    notify_tx: channel::Sender<usize>,
    ragged_block_sender: Arc<Mutex<HashMap<usize, channel::Sender<String>>>>,
}

impl TokenSampler {
    /// 初始化 TokenSampler
    ///
    /// - `tokenizer`：已加载的 Tokenizer
    /// - `tokenizer_config_path`：tokenizer_config.json 路径
    /// - `num_producers`：生产者线程数量
    /// - `channel_capacity`：有界通道容量（缓冲区大小）
    pub fn new(
        tokenizer: Tokenizer,
        tokenizer_config_path: String,
        num_producers: usize,
        channel_capacity: usize,
        block_size: u32,
    ) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true) as u32;

        let data = fs::read_to_string(&tokenizer_config_path)
            .expect("Failed to read tokenizer config file");
        let json: Value =
            serde_json::from_str(&data).expect("Failed to parse tokenizer config file as JSON");

        let splitter = Self::resolve_splitter(&json);

        let ragged_block_cache: Arc<Mutex<HashMap<usize, channel::Receiver<String>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let ragged_block_sender: Arc<Mutex<HashMap<usize, channel::Sender<String>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (notify_tx, notify_rx) = channel::unbounded::<usize>();

        // 有界通道
        let (tx, rx) = channel::bounded::<String>(channel_capacity);
        let rx_arc = Arc::new(Mutex::new(rx));

        // 启动生产者线程
        for i in 0..num_producers {
            let tokenizer_clone = tokenizer.clone();
            let splitter_clone = splitter.clone();
            let notify_rx_clone = notify_rx.clone();
            let ragged_block_sender = Arc::clone(&ragged_block_sender);
            // let bs_clone = Arc::clone(&block_size);
            let tx_clone = tx.clone();

            thread::spawn(move || {
                Self::producer_loop(
                    i,
                    tokenizer_clone,
                    splitter_clone,
                    block_size,
                    tx_clone,
                    notify_rx_clone,
                    ragged_block_sender,
                );
            });
        }

        tracing::info!("Warmup start...");
        for i in 1..block_size {
            let (tx, rx) = channel::bounded::<String>(channel_capacity);
            for _ in 1..20 {
                let prompt = Self::generate_block(&tokenizer, &splitter, i as usize);
                tx.send(prompt).unwrap();
            }
            ragged_block_sender.lock().unwrap().insert(i as usize, tx);
            ragged_block_cache.lock().unwrap().insert(i as usize, rx);
        }
        tracing::info!("Warmup finished!");

        Self {
            tokenizer,
            vocab_size,
            splitter,
            block_size,
            receiver: rx_arc,
            ragged_block_cache,
            notify_tx,
            ragged_block_sender,
        }
    }

    /// 设置 block size（会唤醒生产者）
    // pub fn set_block_size(&self, block_size: u32) {
    //     let mut bs = self.block_size.lock().unwrap();
    //     *bs = block_size;
    // }

    /// 生产者线程逻辑
    fn producer_loop(
        id: usize,
        tokenizer: Tokenizer,
        splitter: Vec<String>,
        block_size: u32,
        tx: channel::Sender<String>,
        notify_rx: channel::Receiver<usize>,
        ragged_block_sender: Arc<Mutex<HashMap<usize, channel::Sender<String>>>>,
    ) {
        let mut local_sample = Vec::new();
        loop {
            // 1️⃣ 优先尝试生成主 block_size 样本
            let sample = if local_sample.is_empty() {
                Self::generate_block(&tokenizer, &splitter, block_size as usize)
            } else {
                local_sample.pop().unwrap()
            };

            // 2️⃣ 尝试发送主样本（不会阻塞）
            match tx.try_send(sample) {
                Ok(_) => continue, // 成功填充 -> 继续下一轮
                Err(channel::TrySendError::Full(x)) => {
                    local_sample.push(x);
                    // 主通道已满 -> 去监听通知
                    match notify_rx.recv_timeout(Duration::from_millis(5)) {
                        Ok(size) => {
                            // 收到通知 -> 生成对应样本并发送到 ragged 通道
                            let ragged_sample = Self::generate_block(&tokenizer, &splitter, size);
                            if let Some(sender) = ragged_block_sender.lock().unwrap().get(&size) {
                                let _ = sender.try_send(ragged_sample);
                            }
                        }
                        Err(channel::RecvTimeoutError::Timeout) => {
                            // 超时 -> 什么都不做，继续下一轮
                            continue;
                        }
                        Err(channel::RecvTimeoutError::Disconnected) => {
                            eprintln!("Producer-{id} notify_rx disconnected, exiting");
                            break;
                        }
                    }
                }
                Err(channel::TrySendError::Disconnected(_)) => {
                    // 主通道关闭 -> 退出
                    eprintln!("Producer-{id} tx disconnected, exiting");
                    break;
                }
            }
        }

        eprintln!("Producer-{id} exited");
    }
    /// 通用 block 生成函数（被 producer 与 gen_string 共用）
    fn generate_block(tokenizer: &Tokenizer, splitter: &[String], n: usize) -> String {
        let mut rng = rand::thread_rng();
        let vocab_size = tokenizer.get_vocab_size(true) as u32;

        // let generate_time = std::time::Instant::now();
        match n {
            0 => return String::new(),
            1 => return splitter[0].clone(),
            2 => {
                return if splitter.len() == 2 {
                    format!("{}{}", splitter[0], splitter[1])
                } else {
                    splitter[0].repeat(2)
                };
            }
            _ => {}
        }

        loop {
            let tokens: Vec<u32> = (0..2 * n).map(|_| rng.gen_range(0..vocab_size)).collect();
            let decoded = tokenizer.decode(&tokens, false).unwrap_or_default();
            let encoded = tokenizer.encode(decoded, false).unwrap();
            let mut ids = encoded.get_ids().to_vec();
            ids.truncate(n.saturating_sub(2));

            let mut result = tokenizer.decode(&ids, false).unwrap_or_default();

            // 加上 splitter
            if splitter.len() == 2 {
                result.insert_str(0, &splitter[0]);
                result.push_str(&splitter[1]);
            } else {
                result.insert_str(0, &splitter[0]);
                result.push_str(&splitter[0]);
            }

            // 验证长度
            let reencoded_len = tokenizer
                .encode(result.clone(), false)
                .unwrap()
                .get_ids()
                .len();
            if reencoded_len == n {
                // let duration = generate_time.elapsed();
                // tracing::info!("Generated block of size {n} in {duration:?}");
                return result;
            }
        }
    }

    /// 解析 tokenizer_config.json 中的特殊 token
    fn resolve_splitter(json: &Value) -> Vec<String> {
        if let Some(pad) = json.get("pad_token").and_then(|v| v.as_str()) {
            vec![pad.to_string()]
        } else {
            let bos = json.get("bos_token").and_then(|v| v.as_str()).unwrap();
            let eos = json.get("eos_token").and_then(|v| v.as_str()).unwrap();
            vec![bos.to_string(), eos.to_string()]
        }
    }

    /// 消费接口
    ///
    /// 若 n == block_size，则尝试从缓存中取出；
    /// 否则即时生成。
    #[instrument(skip_all, fields(block_size = n), level = Level::TRACE)]
    pub fn gen_string(&self, n: usize) -> String {
        // let bs_guard = self.block_size.lock().unwrap();
        // if let Some(bs) = *bs_guard {
        //     if n as u32 == bs {
        // 尝试从 channel 中获取

        if self.block_size == n as u32 {
            if let Ok(sample) = self.receiver.lock().unwrap().recv() {
                return sample;
            }
        }

        if let Some(rx) = self.ragged_block_cache.lock().unwrap().get(&n) {
            if let Ok(sample) = rx.try_recv() {
                self.notify_tx.send(n).unwrap();
                return sample;
            } else {
                self.notify_tx.send(n).unwrap();
                // 通道空，直接生成
                return Self::generate_block(&self.tokenizer, &self.splitter, n);
            }
        } else {
            panic!("No cache for block size {n}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const QWEN2_EOS_TOKEN: u32 = 151645;
    const QWEN2_BOS_TOKEN: u32 = 151644;
    const QWEM2_PAD_TOKEN: u32 = 151643;

    fn validate_block_generation(tokenizer: &Tokenizer, size: usize, stride: usize) -> (String, usize) {
        let mut rng = rand::thread_rng();
        let vocab_size = tokenizer.get_vocab_size(true) as u32;
        let tokens: Vec<u32> = (0..2 * size).map(|_| rng.gen_range(0..vocab_size)).collect();
        let decoded = tokenizer.decode(&tokens, false).unwrap_or_default();
        let encoded = tokenizer.encode(decoded, false).unwrap();
        let mut all_tokens = encoded.get_ids().to_vec();
        all_tokens.truncate(size);

        let mut ret = String::with_capacity(size);
        let mut cnt = 0;
        let mut begin = 0;
        let mut end = begin + stride - 2;

        while end <= all_tokens.len() {
            let mut miss = 0;
            loop {
                let tokens = &all_tokens[begin..end];
                let mut asm_tokens = Vec::with_capacity(stride + miss);
                asm_tokens.push(QWEN2_BOS_TOKEN);
                asm_tokens.extend_from_slice(tokens);
                asm_tokens.push(QWEN2_EOS_TOKEN);
                let result = tokenizer.decode(&asm_tokens, false).unwrap_or_default();
                let reencoded_len = tokenizer
                    .encode(result.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                if reencoded_len < stride {
                    miss += 1;
                    end += 1;
                    if end > all_tokens.len() {
                        break;
                    }
                } else if reencoded_len > stride {
                    // NOTE: non-unit stepping
                    begin = end;
                    end += stride - 2;
                    break;
                } else {
                    begin = end;
                    end += stride - 2;
                    cnt += 1;
                    ret.push_str(&result);
                    break;
                }
            }
        }
        (ret, cnt)
    }

    /// 这个测试测量 encode/decode 随 token 数 n 增加的时延。
    ///
    /// 输出格式：
    /// ```
    /// n=16, time=1.23ms
    /// n=32, time=2.12ms
    /// ...
    /// ```
    #[test]
    fn test_gen_string_latency_scaling() {
        // 加载 tokenizer
        let tokenizer_path = "data/tokenizer.json"; // 你自己的路径
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

        // tokenizer_config.json 路径
        let tokenizer_config_path = "data/tokenizer_config.json".to_string();

        // 测试 token 数范围
        let test_sizes: Vec<usize> = (16..=1024).step_by(32).collect();

        println!("==== TokenSampler decode latency test ====");
        println!("{:<8} | {:<12}", "n", "time (ms)");
        println!("--------------------------------");

        let mut total_cnt = 0;
        let mut total_elapsed = 0.;
        let stride = 16;
        for &n in &test_sizes {
            let start = Instant::now();
            for _ in 0..5 {
                // let _ = generate_block(&tokenizer, n);
                let (result, cnt) = validate_block_generation(&tokenizer, 2048, stride);
                let elapsed = start.elapsed();
                let elapsed_ms = (elapsed.as_secs_f64() * 1000.0 * 100.0).round() / 100.0; // 保留两位小数
                let reencoded_len = tokenizer
                    .encode(result.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                let s = if reencoded_len == cnt * 16 {
                    "OK"
                } else {
                    println!("encode len: {reencoded_len} | expected: {}", cnt * 16);
                    "Err"
                };
                println!("{s}:> {:<8} | {:<12.2}", cnt, elapsed_ms);
                total_cnt += cnt;
                total_elapsed += elapsed_ms;
            }
        }
        println!("--------------------------------");
        println!("Speed: {:<4}ms/block | block size: {stride}", total_elapsed / total_cnt as f64);
    }
}
