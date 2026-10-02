//! Experimental three-scale continuous field for the v10 sibling substrate.
//!
//! This module deliberately does not call the v9 neural CA. Every scale takes
//! a bounded residual step made from directed local transport, diffusion,
//! pointwise reaction, and bidirectional exchange with adjacent scales. All
//! three scales read the same recurrent memory but keep independent state.

use super::*;

pub(super) const COARSE_H: usize = 16;
pub(super) const COARSE_W: usize = 16;
pub(super) const FIELD_DT: [f64; 3] = [1.0, 0.5, 0.25];
const VELOCITY_LIMIT: f64 = 0.10;
const DIFFUSION_LIMIT: f64 = 0.08;
const REACTION_LIMIT: f64 = 0.025;
const EXCHANGE_LIMIT: f64 = 0.025;
const FIELD_LEAK: f64 = 0.020;

struct ScaleRule {
    reaction_in: TensorTransform,
    reaction_out: TensorTransform,
    velocity: TensorTransform,
    diffusion: TensorTransform,
    recurrent_reaction: Linear,
    recurrent_velocity: Linear,
}

impl ScaleRule {
    fn new(vb: VBV) -> Result<Self> {
        Ok(Self {
            reaction_in: conv2d_pointwise(CA_CHANNELS, CA_HIDDEN, vb.pp("reaction_in"))?,
            reaction_out: conv2d_pointwise(CA_HIDDEN, CA_CHANNELS, vb.pp("reaction_out"))?,
            velocity: conv2d_pointwise(CA_CHANNELS, 2, vb.pp("velocity"))?,
            diffusion: conv2d_pointwise(CA_CHANNELS, 1, vb.pp("diffusion"))?,
            recurrent_reaction: candle_nn::linear(
                MEMORY_DIM,
                CA_CHANNELS,
                vb.pp("recurrent_reaction"),
            )?,
            recurrent_velocity: candle_nn::linear(MEMORY_DIM, 2, vb.pp("recurrent_velocity"))?,
        })
    }

    fn rate(&self, x: &Tensor, mem: &Tensor) -> CResult<(Tensor, Tensor, Tensor)> {
        let (batch, channels, height, width) = x.dims4()?;
        if batch != 1 || channels != CA_CHANNELS || mem.dims() != [1, MEMORY_DIM] {
            candle_core::bail!(
                "msfield expected 1x{}xHxW state and 1x{} memory",
                CA_CHANNELS,
                MEMORY_DIM
            );
        }

        // The four shifted fields share one boundary construction. Horizontal
        // crossings use the same Klein seam as the v9 field, so the experiment
        // changes the update rule without also changing its spatial quotient.
        let padded = klein_pad2d(x, 1)?;
        let up = padded.narrow(2, 0, height)?.narrow(3, 1, width)?;
        let down = padded.narrow(2, 2, height)?.narrow(3, 1, width)?;
        let left = padded.narrow(2, 1, height)?.narrow(3, 0, width)?;
        let right = padded.narrow(2, 1, height)?.narrow(3, 2, width)?;

        // Positive x/y velocity pulls content from left/up. Coefficients are
        // nonnegative and at most 0.10 on each axis, keeping this transport
        // term a bounded local interpolation rather than an unconstrained conv.
        let recurrent_velocity = self
            .recurrent_velocity
            .forward(mem)?
            .reshape((1, 2, 1, 1))?;
        let velocity = (self.velocity)(x)?
            .broadcast_add(&recurrent_velocity)?
            .tanh()?
            .affine(VELOCITY_LIMIT, 0.0)?;
        let vx = velocity.narrow(1, 0, 1)?;
        let vy = velocity.narrow(1, 1, 1)?;
        let adv_x = left.sub(x)?.broadcast_mul(&vx.relu()?)?.add(
            &right
                .sub(x)?
                .broadcast_mul(&vx.affine(-1.0, 0.0)?.relu()?)?,
        )?;
        let adv_y = up
            .sub(x)?
            .broadcast_mul(&vy.relu()?)?
            .add(&down.sub(x)?.broadcast_mul(&vy.affine(-1.0, 0.0)?.relu()?)?)?;
        let advection = adv_x.add(&adv_y)?;

        // A positive, learned, spatially varying diffusion coefficient. The
        // four-neighbor average is a convex operator when used at this bound.
        let diffusivity =
            candle_nn::ops::sigmoid(&(self.diffusion)(x)?)?.affine(DIFFUSION_LIMIT, 0.0)?;
        let neighbor_mean = up.add(&down)?.add(&left)?.add(&right)?.affine(0.25, 0.0)?;
        let diffusive = neighbor_mean.sub(x)?.broadcast_mul(&diffusivity)?;

        let recurrent_reaction =
            self.recurrent_reaction
                .forward(mem)?
                .reshape((1, CA_CHANNELS, 1, 1))?;
        let reaction =
            (self.reaction_out)(&candle_nn::Activation::Swish.forward(&(self.reaction_in)(x)?)?)?
                .broadcast_add(&recurrent_reaction)?
                .tanh()?
                .affine(REACTION_LIMIT, 0.0)?;

        let rate = advection
            .add(&diffusive)?
            .add(&reaction)?
            .sub(&x.affine(FIELD_LEAK, 0.0)?)?;
        Ok((
            rate,
            velocity.detach().abs()?.mean_all()?,
            diffusivity.detach().mean_all()?,
        ))
    }
}

struct Exchange {
    project: TensorTransform,
    gate_logit: Tensor,
}

impl Exchange {
    fn new(vb: VBV) -> Result<Self> {
        Ok(Self {
            project: conv2d_pointwise(CA_CHANNELS, CA_CHANNELS, vb.pp("project"))?,
            gate_logit: vb.get_with_hints(
                (1, 1, 1, 1),
                "gate_logit",
                candle_nn::Init::Const(-2.0),
            )?,
        })
    }

    fn residual(&self, source_at_target_scale: &Tensor, target: &Tensor) -> CResult<Tensor> {
        let projected = (self.project)(source_at_target_scale)?.tanh()?;
        let gain = candle_nn::ops::sigmoid(&self.gate_logit)?.affine(EXCHANGE_LIMIT, 0.0)?;
        projected.sub(target)?.broadcast_mul(&gain)
    }
}

/// The canonical renderer still consumes the 64x64 and 32x32 fields; the
/// 16x16 state reaches it through learned meso/coarse exchange.
pub(super) struct MsField {
    pub(super) analysis_disable_exchange: bool,
    // A tensor-name discriminator in SafeTensors. The v9 loader must reject
    // this marker before its permissive compatible-tensor migration begins.
    _schema_marker: Tensor,
    fine: ScaleRule,
    meso: ScaleRule,
    coarse: ScaleRule,
    meso_to_fine: Exchange,
    fine_to_meso: Exchange,
    coarse_to_meso: Exchange,
    meso_to_coarse: Exchange,
}

pub(super) struct MsFieldDiagnostics {
    pub(super) velocity_mean_abs: [Tensor; 3],
    pub(super) diffusivity_mean: [Tensor; 3],
}

impl MsField {
    pub(super) fn new(vb: VBV, _device: &Device) -> Result<Self> {
        Ok(Self {
            analysis_disable_exchange: false,
            _schema_marker: vb.get_with_hints(
                (1,),
                "schema_marker",
                candle_nn::Init::Const(10.0),
            )?,
            fine: ScaleRule::new(vb.pp("fine"))?,
            meso: ScaleRule::new(vb.pp("meso"))?,
            coarse: ScaleRule::new(vb.pp("coarse"))?,
            meso_to_fine: Exchange::new(vb.pp("meso_to_fine"))?,
            fine_to_meso: Exchange::new(vb.pp("fine_to_meso"))?,
            coarse_to_meso: Exchange::new(vb.pp("coarse_to_meso"))?,
            meso_to_coarse: Exchange::new(vb.pp("meso_to_coarse"))?,
        })
    }

    pub(super) fn step(
        &self,
        fine: &Tensor,
        meso: &Tensor,
        coarse: &Tensor,
        mem: &Tensor,
    ) -> CResult<(Tensor, Tensor, Tensor, MsFieldDiagnostics)> {
        if fine.dims() != [1, CA_CHANNELS, GRID_H, GRID_W]
            || meso.dims() != [1, CA_CHANNELS, MACRO_H, MACRO_W]
            || coarse.dims() != [1, CA_CHANNELS, COARSE_H, COARSE_W]
        {
            candle_core::bail!("msfield state dimensions do not match fine/meso/coarse topology");
        }

        let (fine_rate, fine_velocity, fine_diffusion) = self.fine.rate(fine, mem)?;
        let (meso_rate, meso_velocity, meso_diffusion) = self.meso.rate(meso, mem)?;
        let (coarse_rate, coarse_velocity, coarse_diffusion) = self.coarse.rate(coarse, mem)?;

        // Exchange is simultaneous: every projection reads the pre-step state.
        // This prevents one scale from seeing another scale's already-updated
        // state within the same chunk.
        let (fine_rate, meso_rate, coarse_rate) = if self.analysis_disable_exchange {
            (fine_rate, meso_rate, coarse_rate)
        } else {
            let meso_up = meso.upsample_nearest2d(GRID_H, GRID_W)?;
            let fine_down = decimate2_2d(fine)?;
            let coarse_up = coarse.upsample_nearest2d(MACRO_H, MACRO_W)?;
            let meso_down = decimate2_2d(meso)?;
            let fine_rate = fine_rate.add(&self.meso_to_fine.residual(&meso_up, fine)?)?;
            let meso_rate = meso_rate
                .add(&self.fine_to_meso.residual(&fine_down, meso)?)?
                .add(&self.coarse_to_meso.residual(&coarse_up, meso)?)?;
            let coarse_rate =
                coarse_rate.add(&self.meso_to_coarse.residual(&meso_down, coarse)?)?;
            (fine_rate, meso_rate, coarse_rate)
        };

        // Explicit Euler with fixed scale clocks. A hard state rail is a final
        // numerical guard, not a proof that the learned map is contractive.
        let next_fine = fine
            .add(&fine_rate.affine(FIELD_DT[0], 0.0)?)?
            .clamp(-1.0f32, 1.0f32)?;
        let next_meso = meso
            .add(&meso_rate.affine(FIELD_DT[1], 0.0)?)?
            .clamp(-1.0f32, 1.0f32)?;
        let next_coarse = coarse
            .add(&coarse_rate.affine(FIELD_DT[2], 0.0)?)?
            .clamp(-1.0f32, 1.0f32)?;

        Ok((
            next_fine,
            next_meso,
            next_coarse,
            MsFieldDiagnostics {
                velocity_mean_abs: [fine_velocity, meso_velocity, coarse_velocity],
                diffusivity_mean: [fine_diffusion, meso_diffusion, coarse_diffusion],
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msfield_step_is_finite_and_differentiable() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let vb = VBV::from_varmap(&vars, DType::F32, &device);
        let field = MsField::new(vb.pp("msfield"), &device)?;
        deterministic_reinit(&vars, 42, &device)?;
        let mut rng = RuntimeRng::seed_from_u64(123);
        let fine = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 0.1, &device)?;
        let meso = randn_t(&mut rng, &[1, CA_CHANNELS, MACRO_H, MACRO_W], 0.1, &device)?;
        let coarse = randn_t(
            &mut rng,
            &[1, CA_CHANNELS, COARSE_H, COARSE_W],
            0.1,
            &device,
        )?;
        let mem = randn_t(&mut rng, &[1, MEMORY_DIM], 0.1, &device)?;
        let (fine_next, meso_next, coarse_next, diagnostics) =
            field.step(&fine, &meso, &coarse, &mem)?;
        for value in [&fine_next, &meso_next, &coarse_next] {
            let flat = value.flatten_all()?.to_vec1::<f32>()?;
            assert!(flat.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        }
        for value in diagnostics
            .velocity_mean_abs
            .iter()
            .chain(diagnostics.diffusivity_mean.iter())
        {
            assert!(value.to_scalar::<f32>()?.is_finite());
        }
        let loss = fine_next
            .sqr()?
            .mean_all()?
            .add(&meso_next.sqr()?.mean_all()?)?
            .add(&coarse_next.sqr()?.mean_all()?)?;
        let grads = loss.backward()?;
        let data = vars.data().lock().unwrap();
        let nonzero_finite = data
            .iter()
            .filter(|(_, var)| {
                grads
                    .get(var.as_tensor())
                    .and_then(|g| g.flatten_all().ok())
                    .and_then(|g| g.to_vec1::<f32>().ok())
                    .is_some_and(|g| {
                        g.iter().all(|x| x.is_finite()) && g.iter().any(|x| x.abs() > 0.0)
                    })
            })
            .count();
        assert!(
            nonzero_finite > 0,
            "msfield must backpropagate into its learned rule"
        );
        Ok(())
    }

    #[test]
    fn msfield_seed_repeats_exact_first_step() -> Result<()> {
        fn first_step(seed: u64) -> Result<Vec<f32>> {
            let device = Device::Cpu;
            let vars = VarMap::new();
            let vb = VBV::from_varmap(&vars, DType::F32, &device);
            let field = MsField::new(vb.pp("msfield"), &device)?;
            deterministic_reinit(&vars, seed, &device)?;
            let mut rng = RuntimeRng::seed_from_u64(123);
            let fine = randn_t(&mut rng, &[1, CA_CHANNELS, GRID_H, GRID_W], 0.1, &device)?;
            let meso = randn_t(&mut rng, &[1, CA_CHANNELS, MACRO_H, MACRO_W], 0.1, &device)?;
            let coarse = randn_t(
                &mut rng,
                &[1, CA_CHANNELS, COARSE_H, COARSE_W],
                0.1,
                &device,
            )?;
            let mem = randn_t(&mut rng, &[1, MEMORY_DIM], 0.1, &device)?;
            let (next, _, _, _) = field.step(&fine, &meso, &coarse, &mem)?;
            Ok(next.flatten_all()?.to_vec1::<f32>()?)
        }
        assert_eq!(first_step(7)?, first_step(7)?);
        Ok(())
    }

    #[test]
    fn coarse_state_reaches_meso_then_fine() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let vb = VBV::from_varmap(&vars, DType::F32, &device);
        let field = MsField::new(vb.pp("msfield"), &device)?;
        deterministic_reinit(&vars, 19, &device)?;
        let fine = Tensor::zeros((1, CA_CHANNELS, GRID_H, GRID_W), DType::F32, &device)?;
        let meso = Tensor::zeros((1, CA_CHANNELS, MACRO_H, MACRO_W), DType::F32, &device)?;
        let zero = Tensor::zeros((1, CA_CHANNELS, COARSE_H, COARSE_W), DType::F32, &device)?;
        let impulse = Tensor::ones((1, CA_CHANNELS, COARSE_H, COARSE_W), DType::F32, &device)?
            .affine(0.5, 0.0)?;
        let memory = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)?;
        let (fine_a, meso_a, coarse_a, _) = field.step(&fine, &meso, &zero, &memory)?;
        let (fine_b, meso_b, coarse_b, _) = field.step(&fine, &meso, &impulse, &memory)?;
        let meso_difference = meso_a.sub(&meso_b)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(meso_difference > 1e-7, "coarse must reach meso");
        let (fine_aa, _, _, _) = field.step(&fine_a, &meso_a, &coarse_a, &memory)?;
        let (fine_bb, _, _, _) = field.step(&fine_b, &meso_b, &coarse_b, &memory)?;
        let fine_difference = fine_aa
            .sub(&fine_bb)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        assert!(
            fine_difference > 1e-9,
            "coarse must reach fine through meso"
        );
        Ok(())
    }

    #[test]
    fn late_fine_loss_reaches_earlier_coarse_state_only_inside_tape() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let vb = VBV::from_varmap(&vars, DType::F32, &device);
        let field = MsField::new(vb.pp("msfield"), &device)?;
        deterministic_reinit(&vars, 19, &device)?;
        let fine = Tensor::zeros((1, CA_CHANNELS, GRID_H, GRID_W), DType::F32, &device)?;
        let meso = Tensor::zeros((1, CA_CHANNELS, MACRO_H, MACRO_W), DType::F32, &device)?;
        let coarse = Var::from_tensor(
            &Tensor::ones((1, CA_CHANNELS, COARSE_H, COARSE_W), DType::F32, &device)?
                .affine(0.05, 0.0)?,
        )?;
        let memory = Tensor::zeros((1, MEMORY_DIM), DType::F32, &device)?;
        let (fine_1, meso_1, coarse_1, _) =
            field.step(&fine, &meso, coarse.as_tensor(), &memory)?;
        let (fine_2, _, _, _) = field.step(&fine_1, &meso_1, &coarse_1, &memory)?;
        let within = fine_2.sqr()?.mean_all()?.backward()?;
        let gradient = within
            .get(coarse.as_tensor())
            .ok_or_else(|| anyhow::anyhow!("earlier coarse state received no late fine gradient"))?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(gradient.is_finite() && gradient > 0.0);

        let (fine_after, _, _, _) = field.step(
            &fine_1.detach(),
            &meso_1.detach(),
            &coarse_1.detach(),
            &memory,
        )?;
        let beyond = fine_after.sqr()?.mean_all()?.backward()?;
        assert!(beyond.get(coarse.as_tensor()).is_none());
        Ok(())
    }
}
