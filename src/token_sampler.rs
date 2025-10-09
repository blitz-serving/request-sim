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

        (0..n)
            .map(|i| {
                if i == n - 1 {
                    tokenizer
                        .decode(&[self.splitter], false)
                        .unwrap_or_default()
                } else {
                    let id = rng.gen_range(0..vocab_size);
                    let dec_all = tokenizer.decode(&[id], false).unwrap_or_default();
                    let dec_skip = tokenizer.decode(&[id], true).unwrap_or_default();
                    if !dec_all.is_empty() && dec_all.len() == dec_skip.len() {
                        dec_all
                    } else {
                        "Alice".to_string()
                    }
                }
            })
            .collect()
    }
}
