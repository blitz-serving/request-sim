use rand::Rng;
use tokenizers::Tokenizer;

pub struct TokenSampler {
    tokenizer: Tokenizer,
    vocab_size: u32,
    splitter: u32,
}

impl TokenSampler {
    pub fn new(tokenizer: Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true) as u32;
        let splitter = *tokenizer.get_added_tokens_decoder().keys().next().unwrap();
        Self {
            tokenizer,
            vocab_size,
            splitter,
        }
    }

    pub fn gen_string(&self, n: usize) -> String {
        if n == 0 {
            return String::new();
        }

        let mut rng = rand::thread_rng();
        let tokenizer = &self.tokenizer;
        let vocab_size = self.vocab_size;

        let mut result;
        loop {
            // 1️⃣ 生成 2n 个随机 token
            let tokens: Vec<u32> = (0..2 * n).map(|_| rng.gen_range(0..vocab_size)).collect();

            // 2️⃣ decode → encode
            let decoded = tokenizer.decode(&tokens, false).unwrap_or_default();
            let encoded = tokenizer.encode(decoded, false).unwrap();
            let mut ids = encoded.get_ids().to_vec();

            // 3️⃣ 截取前 n-1 个
            ids.truncate(n.saturating_sub(2));

            // 4️⃣ 加上 splitter
            ids.push(self.splitter);
            ids.insert(0, self.splitter);

            // 5️⃣ decode 最终结果
            result = tokenizer.decode(&ids, false).unwrap_or_default();

            // 6️⃣ 验证：重新 encode 后长度是否等于 n
            let reencoded_len = tokenizer
                .encode(result.clone(), false)
                .unwrap()
                .get_ids()
                .len();

            if reencoded_len == n {
                // tracing::info!("Generated string of length {}: {:?}", n, ids);
                break;
            }

            // 否则继续循环
            // （一般 1~3 次就能找到符合的结果）
        }
        return result;
    }
}
