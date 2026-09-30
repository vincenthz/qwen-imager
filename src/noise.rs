use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

/// Channel-first standard Gaussian noise. Shared-grid pooling divides each
/// block's sum by sqrt(block area), preserving unit variance and independence
/// between non-overlapping output cells. Request validation checks dimensions.
pub(crate) fn sample(seed: u64, height: usize, width: usize, source: Option<usize>) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let (sh, sw) = source.map_or((height, width), |side| (side, side));
    let high: Vec<f32> = (0..64 * sh * sw)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    if (sh, sw) == (height, width) {
        return high;
    }
    let factor = sh / height;
    let mut low = vec![0.; 64 * height * width];
    for channel in 0..64 {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.;
                for dy in 0..factor {
                    for dx in 0..factor {
                        sum += high[channel * sh * sw + (y * factor + dy) * sw + x * factor + dx];
                    }
                }
                low[channel * height * width + y * width + x] = sum / factor as f32;
            }
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_noise_pools_the_same_high_resolution_seed_per_channel() {
        let high = sample(42, 128, 128, None);
        assert_eq!(high, sample(42, 128, 128, Some(128)));
        let low = sample(42, 32, 32, Some(128));
        for channel in [0, 1, 63] {
            for (y, x) in [(0, 0), (7, 11), (31, 31)] {
                let expected: f32 = (0..4)
                    .flat_map(|dy| (0..4).map(move |dx| (dy, dx)))
                    .map(|(dy, dx)| high[channel * 128 * 128 + (y * 4 + dy) * 128 + x * 4 + dx])
                    .sum::<f32>()
                    / 4.;
                assert_eq!(low[channel * 32 * 32 + y * 32 + x], expected);
            }
        }
        let mean = low.iter().sum::<f32>() / low.len() as f32;
        let variance = low.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / low.len() as f32;
        assert!(mean.abs() < 0.02);
        assert!((variance - 1.).abs() < 0.03);
    }

    #[test]
    fn native_noise_preserves_the_original_rng_sequence() {
        let mut rng = StdRng::seed_from_u64(7);
        let original: Vec<f32> = (0..64 * 32 * 32)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        assert_eq!(sample(7, 32, 32, None), original);
    }
}
