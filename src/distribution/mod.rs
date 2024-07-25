pub mod gamma;

pub trait Distribution: Send + Sync {
    fn generate(&self) -> f64;
}

impl Distribution for Box<dyn Distribution> {
    fn generate(&self) -> f64 {
        self.as_ref().generate()
    }
}
