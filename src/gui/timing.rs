use std::time::{Duration, Instant};

/// Measure full steps (denoising plus preview decoding). Model loading is a
/// one-off cost: show it in elapsed time, but don't multiply it by future steps.
pub struct GenerationTiming {
    started: Instant,
    steps_started: Option<Instant>,
    last_preview: Option<Instant>,
    completed: usize,
    total: usize,
}

impl GenerationTiming {
    pub fn new(total: usize) -> Self {
        Self {
            started: Instant::now(),
            steps_started: None,
            last_preview: None,
            completed: 0,
            total,
        }
    }

    pub fn start_steps(&mut self) {
        self.steps_started.get_or_insert_with(Instant::now);
    }

    pub fn complete_step(&mut self, step: usize) {
        self.completed = step.min(self.total);
        self.last_preview = Some(Instant::now());
    }

    pub fn label(&self) -> String {
        let now = Instant::now();
        let elapsed = format_duration(now.duration_since(self.started));
        let remaining = match (self.steps_started, self.last_preview) {
            (Some(start), Some(last)) if self.completed > 0 => {
                let average = last.duration_since(start).as_secs_f64() / self.completed as f64;
                let predicted = average * (self.total - self.completed) as f64;
                // Account for work since the last preview, without showing a
                // negative countdown when a step takes longer than the average.
                let seconds = (predicted - now.duration_since(last).as_secs_f64()).max(0.);
                if self.completed == self.total {
                    "Finishing…".into()
                } else if seconds < 1. {
                    "Less than 1s remaining (estimated)".into()
                } else {
                    format!(
                        "~{} remaining",
                        format_duration(Duration::from_secs_f64(seconds))
                    )
                }
            }
            _ => "Estimating remaining time…".into(),
        };
        format!("Elapsed {elapsed} · {remaining}")
    }
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3600 {
        format!(
            "{}h {:02}m {:02}s",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}
