use super::Distribution;

#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    mean: f64,
    cv: f64,
    gamma: rand_distr::Gamma<f64>,
}

impl Gamma {
    pub fn new(mean: f64, cv: f64) -> Self {
        let shape = 1.0 / cv / cv;
        let scale = mean * cv * cv;
        let gamma = rand_distr::Gamma::new(shape, scale).unwrap();
        Self { mean, cv, gamma }
    }

    pub fn info(&self) -> String {
        format!("Gamma(mean={}, cv={})", self.mean, self.cv,)
    }
}

impl Distribution for Gamma {
    fn generate(&self) -> f64 {
        rand_distr::Distribution::sample(&self.gamma, &mut rand::thread_rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma_mean(sample_len: usize, mean: f64, cv: f64) -> f64 {
        let gamma = Gamma::new(mean, cv);
        let sampled = (0..sample_len)
            .map(|_| gamma.generate())
            .collect::<Vec<f64>>();
        let sampled_mean = sampled.iter().sum::<f64>() / (sample_len as f64);
        sampled_mean
    }

    fn gamma_cv(sample_len: usize, mean: f64, cv: f64) -> f64 {
        let gamma = Gamma::new(mean, cv);
        let sampled = (0..sample_len)
            .map(|_| gamma.generate())
            .collect::<Vec<f64>>();
        let sampled_mean = sampled.iter().sum::<f64>() / (sample_len as f64);
        let sampled_var = sampled
            .iter()
            .map(|x| (x - sampled_mean).powi(2))
            .sum::<f64>()
            / (sample_len as f64);
        let sampled_cv = sampled_var.sqrt() / sampled_mean;
        sampled_cv
    }

    #[test]
    fn test_gamma_mean() {
        let sample_len = 100_000;
        let mean = 100.0;
        let cv = 0.5;
        for _ in 0..10 {
            let sampled_mean = gamma_mean(sample_len, mean, cv);
            assert!((sampled_mean - mean).abs() / mean < 0.01);
        }
    }

    #[test]
    fn test_gamma_cv() {
        let sample_len = 100_000;
        let mean = 100.0;
        let cv = 0.5;
        for _ in 0..10 {
            let sampled_cv = gamma_cv(sample_len, mean, cv);
            assert!((sampled_cv - cv).abs() / cv < 0.01);
        }
    }
}
