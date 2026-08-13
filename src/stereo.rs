use candle_core::{Result, Tensor};

pub const GLOBAL_PAN_LIMIT: f64 = 0.10;
pub const PAN_CENTER_LOSS_WEIGHT: f64 = 0.10;
pub const WIDTH_CONTROL_MIN: f64 = 0.05;
pub const WIDTH_CONTROL_MAX: f64 = 0.50;
pub const WIDTH_CONTROL_SOFTNESS: f64 = 1.0;

pub fn side_gain(last_pan: f32, width_mult: f32, maximum: f32) -> f32 {
    ((1.0 + last_pan.abs() * 0.8) * width_mult.clamp(0.5, 1.6)).clamp(0.5, maximum)
}

pub fn soft_global_pan(raw: &Tensor) -> Result<Tensor> {
    // z / sqrt(1 + z^2) is smooth and bounded like tanh, but its gradient
    // decays polynomially instead of exponentially when an old checkpoint has
    // driven the residual pan head far into saturation.
    let scale = raw.sqr()?.affine(1.0, 1.0)?.sqrt()?;
    raw.div(&scale)?.affine(GLOBAL_PAN_LIMIT, 0.0)
}

pub fn pan_center_loss(pan: &Tensor) -> Result<Tensor> {
    // Penalize only the bounded audible residual. An earlier penalty on the
    // raw coordinate could dominate the complete loss when an old checkpoint
    // had already driven its pan head far into saturation.
    pan.sqr()
}

pub fn soft_width_control(raw: &Tensor) -> Result<Tensor> {
    // Preserve the width head's tensor layout while replacing its saturating
    // sigmoid. The normalized head input keeps the useful operating region
    // near zero, while the polynomial tail retains a recovery gradient and a
    // 0.50 ceiling prevents learned side gain from forcing anti-correlation.
    let unit = raw.div(
        &raw.sqr()?
            .affine(1.0, WIDTH_CONTROL_SOFTNESS * WIDTH_CONTROL_SOFTNESS)?
            .sqrt()?,
    )?;
    let half_range = (WIDTH_CONTROL_MAX - WIDTH_CONTROL_MIN) * 0.5;
    let midpoint = (WIDTH_CONTROL_MAX + WIDTH_CONTROL_MIN) * 0.5;
    unit.affine(half_range, midpoint)
}

pub fn regional_pan_position(partial: usize, columns: usize) -> f32 {
    let column = partial % columns;
    column as f32 / (columns - 1) as f32 * 2.0 - 1.0
}

pub fn correlation_aware_width(side_energy_width: f32, stereo_corr: f32) -> f32 {
    let incoherence = ((1.0 - stereo_corr.clamp(-1.0, 1.0)) * 0.5).sqrt();
    (side_energy_width * incoherence).clamp(0.0, 1.0)
}
