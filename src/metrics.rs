pub struct SystemMetrics {
    pub generate_time: Option<u64>,
    pub get_prompt_time: Option<u64>,
    pub sample_time: Option<u64>,
    pub inflate_time: Option<u64>,
    pub send_gap: Option<u64>,
    pub prev_sample_time: Option<u64>,
    pub post_sample_time: Option<u64>,
}
