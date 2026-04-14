use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::dataset::{LLMTrace, PromptPayload};
#[cfg(feature = "prompt-text-hashed")]
use crate::token_sampler::TokenSampler;

/// A single pre-generated prompt with its metadata.
pub struct CachedPrompt {
    pub prompt: PromptPayload,
    pub input_length: u64,
    pub output_length: u64,
}

/// Pre-generated prompt cache backed by a binary file.
///
/// Binary format (all integers little-endian):
/// ```text
/// [count: u64]
/// For each entry:
///   [input_length: u64]
///   [output_length: u64]
///   [prompt_byte_len: u64]
///   [prompt_bytes: u8 * prompt_byte_len]
/// ```
pub struct PromptCache {
    entries: Vec<CachedPrompt>,
}

impl PromptCache {
    pub fn get(&self, index: usize) -> &CachedPrompt {
        &self.entries[index]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Load cache from file if it exists and has the expected count,
    /// otherwise generate from dataset and write to file.
    #[cfg(feature = "prompt-text-hashed")]
    pub fn load_or_generate(
        dataset: &dyn LLMTrace,
        token_sampler: &TokenSampler,
        cache_path: &Path,
    ) -> Self {
        let expected = dataset.len();

        if cache_path.exists() {
            match Self::load(cache_path) {
                Ok(cache) if cache.len() == expected => {
                    tracing::info!(
                        "Loaded prompt cache from {} ({} entries)",
                        cache_path.display(),
                        cache.len()
                    );
                    return cache;
                }
                Ok(cache) => {
                    tracing::warn!(
                        "Cache count mismatch (file={}, dataset={}), regenerating",
                        cache.len(),
                        expected
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to load cache: {e}, regenerating");
                }
            }
        }

        let cache = Self::generate(dataset, token_sampler);
        if let Err(e) = cache.write(cache_path) {
            tracing::error!("Failed to write cache to {}: {e}", cache_path.display());
        } else {
            tracing::info!(
                "Wrote prompt cache to {} ({} entries)",
                cache_path.display(),
                cache.len()
            );
        }
        cache
    }

    /// Load cache from file if it exists and has the expected count,
    /// otherwise generate from dataset and write to file.
    #[cfg(feature = "prompt-text-plain")]
    pub fn load_or_generate(
        dataset: &dyn LLMTrace,
        cache_path: &Path,
    ) -> Self {
        let expected = dataset.len();

        if cache_path.exists() {
            match Self::load(cache_path) {
                Ok(cache) if cache.len() == expected => {
                    tracing::info!(
                        "Loaded prompt cache from {} ({} entries)",
                        cache_path.display(),
                        cache.len()
                    );
                    return cache;
                }
                Ok(cache) => {
                    tracing::warn!(
                        "Cache count mismatch (file={}, dataset={}), regenerating",
                        cache.len(),
                        expected
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to load cache: {e}, regenerating");
                }
            }
        }

        let cache = Self::generate(dataset);
        if let Err(e) = cache.write(cache_path) {
            tracing::error!("Failed to write cache to {}: {e}", cache_path.display());
        } else {
            tracing::info!(
                "Wrote prompt cache to {} ({} entries)",
                cache_path.display(),
                cache.len()
            );
        }
        cache
    }

    /// Generate all prompts by iterating the dataset.
    #[cfg(feature = "prompt-text-hashed")]
    fn generate(dataset: &dyn LLMTrace, token_sampler: &TokenSampler) -> Self {
        let n = dataset.len();
        tracing::info!("Pre-generating {n} prompts...");
        let start = std::time::Instant::now();

        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let (prompt, input_length, output_length) = dataset.inflate(i, token_sampler);
            entries.push(CachedPrompt {
                prompt,
                input_length,
                output_length,
            });
            if (i + 1) % 10000 == 0 {
                tracing::info!("  pre-gen progress: {}/{n}", i + 1);
            }
        }

        tracing::info!(
            "Pre-generated {n} prompts in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Self { entries }
    }

    /// Generate all prompts by iterating the dataset (plain-text mode).
    #[cfg(feature = "prompt-text-plain")]
    fn generate(dataset: &dyn LLMTrace) -> Self {
        let n = dataset.len();
        tracing::info!("Pre-generating {n} prompts...");
        let start = std::time::Instant::now();

        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let (prompt, input_length, output_length) = dataset.inflate(i);
            entries.push(CachedPrompt {
                prompt,
                input_length,
                output_length,
            });
            if (i + 1) % 10000 == 0 {
                tracing::info!("  pre-gen progress: {}/{n}", i + 1);
            }
        }

        tracing::info!(
            "Pre-generated {n} prompts in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Self { entries }
    }

    /// Load cache from a binary file.
    fn load(path: &Path) -> std::io::Result<Self> {
        let file = fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            reader.read_exact(&mut buf8)?;
            let input_length = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf8)?;
            let output_length = u64::from_le_bytes(buf8);

            let mut tag = [0u8; 1];
            reader.read_exact(&mut tag)?;

            reader.read_exact(&mut buf8)?;
            let prompt_len = u64::from_le_bytes(buf8) as usize;

            let mut prompt_bytes = vec![0u8; prompt_len];
            reader.read_exact(&mut prompt_bytes)?;

            let payload_str = String::from_utf8(prompt_bytes).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;

            let prompt = match tag[0] {
                0 => PromptPayload::Content(payload_str),
                1 => {
                    let val: serde_json::Value = serde_json::from_str(&payload_str)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    PromptPayload::Messages(val)
                }
                t => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unknown PromptPayload tag: {t}"),
                    ));
                }
            };

            entries.push(CachedPrompt {
                prompt,
                input_length,
                output_length,
            });
        }

        Ok(Self { entries })
    }

    /// Write cache to a binary file.
    fn write(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&(self.entries.len() as u64).to_le_bytes())?;
        for entry in &self.entries {
            writer.write_all(&entry.input_length.to_le_bytes())?;
            writer.write_all(&entry.output_length.to_le_bytes())?;

            let (tag, bytes): (u8, Vec<u8>) = match &entry.prompt {
                PromptPayload::Content(s) => (0, s.as_bytes().to_vec()),
                PromptPayload::Messages(v) => (1, serde_json::to_vec(v).unwrap()),
            };
            writer.write_all(&[tag])?;
            writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
            writer.write_all(&bytes)?;
        }
        writer.flush()?;

        Ok(())
    }
}
