use std::time::{Duration, Instant};

/// Measure full automatic-preview steps, or sampling steps in manual mode. Model loading is a
/// one-off cost: show it in elapsed time, but don't multiply it by future steps.
pub struct GenerationTiming {
    started: Instant,
    steps_started: Option<Instant>,
    last_completed: Option<Instant>,
    completed: usize,
    total: usize,
    paused: Option<Instant>,
}

impl GenerationTiming {
    pub fn new(total: usize) -> Self {
        Self {
            started: Instant::now(),
            steps_started: None,
            last_completed: None,
            completed: 0,
            total,
            paused: None,
        }
    }

    /// Freezes the clock while paused; resuming shifts the recorded instants
    /// forward so paused time counts neither as elapsed nor as step time.
    pub fn set_paused(&mut self, paused: bool) {
        match (paused, self.paused) {
            (true, None) => self.paused = Some(Instant::now()),
            (false, Some(since)) => {
                let pause = since.elapsed();
                self.started += pause;
                for instant in [&mut self.steps_started, &mut self.last_completed]
                    .into_iter()
                    .flatten()
                {
                    *instant += pause;
                }
                self.paused = None;
            }
            _ => {}
        }
    }

    fn now(&self) -> Instant {
        self.paused.unwrap_or_else(Instant::now)
    }

    pub fn start_steps(&mut self) {
        let now = self.now();
        self.steps_started.get_or_insert(now);
    }

    pub fn complete_step(&mut self, step: usize) {
        self.completed = step.min(self.total);
        self.last_completed = Some(self.now());
    }

    pub fn label(&self) -> String {
        let now = self.now();
        let elapsed = format_duration(now.duration_since(self.started));
        let remaining = match (self.steps_started, self.last_completed) {
            (Some(start), Some(last)) if self.completed > 0 => {
                let average = last.duration_since(start).as_secs_f64() / self.completed as f64;
                let predicted = average * (self.total - self.completed) as f64;
                // Account for work since the last completed step, without showing a
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
