//! Text written by the checkpoint's Qwen3-VL-8B language model: the text
//! encoder's layers followed by its final norm and LM head.
use crate::{
    ASPECT_RATIOS, Event, Observer, Stage, TextRequest, TextTask, ops, text,
    vision::Visual,
    weights::{Weights, cached_builder},
};
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use image::RgbaImage;
use rand::{Rng, SeedableRng, rngs::StdRng};

const VOCABULARY: usize = 151936;
/// `<|endoftext|>` and `<|im_end|>`.
const STOP: [u32; 2] = [151643, 151645];
/// Applied once to every token already written, against repetitive loops.
const PRESENCE_PENALTY: f32 = 1.0;
const TOP_K: usize = 20;
const TOP_P: f32 = 0.8;

const DESCRIBE: &str = "Describe this image in one detailed paragraph of about 100 to 200 \
words: the main subjects and their appearance, the setting, composition and framing, lighting, \
colors, style or medium, and any visible text, quoted exactly.";
const DESCRIBE_MANY: &str = "Describe each picture in one detailed paragraph of about 100 to 200 \
words: the main subjects and their appearance, the setting, composition and framing, lighting, \
colors, style or medium, and any visible text, quoted exactly.";

const ENHANCE: &str = "You write prompts for a text-to-image model. Rewrite the user's request \
as one detailed English image prompt of about 80 to 150 words. Keep every subject, attribute, \
count, relationship and stated style from the request; do not add anything that contradicts it. \
Add concrete visual detail: the subjects' appearance, the setting, composition and camera \
framing, lighting, color palette, materials, and the style or medium (choose a fitting one if \
none is given). Text that should appear in the image must be quoted exactly, in its original \
language. Never answer or comment on the request, only rewrite it. Reply with JSON only, in the \
form {\"prompt\": \"...\", \"ratio\": \"16:9\"}, where ratio is the aspect ratio that best suits \
the composition, one of 1:1, 4:3, 3:4, 3:2, 2:3, 16:9 or 9:16.";

const ENHANCE_EDIT: &str = "You write instructions for an image-editing model. The user gives \
one or more input images and a request. Look at the images and rewrite the request as one \
precise English edit instruction. Identify what to change by its visible appearance and \
location, say exactly how it changes, and say what must stay the same (such as identity, pose, \
background, lighting and style) unless the request changes it. Refer to inputs as image 1, \
image 2, and so on. Text that should appear in the image must be quoted exactly, in its original \
language. Resolve vague or contradictory requests with the most reasonable interpretation, and \
keep the request's intent. Never answer or comment on the request, only rewrite it. Reply with \
JSON only, in the form {\"prompt\": \"...\"}.";

pub struct Written {
    pub text: String,
    pub ratio: Option<String>,
    pub tokens: usize,
}

/// The chat for `request`, with one `<|image_pad|>` placeholder per image.
fn chat(request: &TextRequest) -> Result<String> {
    let images = request.images.len();
    let prompt = request.prompt.trim();
    let system = match request.task {
        TextTask::Describe => "You are a helpful assistant.",
        TextTask::EnhancePrompt if images == 0 => ENHANCE,
        TextTask::EnhancePrompt => ENHANCE_EDIT,
    };
    let mut user = String::new();
    for i in 0..images {
        let label = match request.task {
            TextTask::EnhancePrompt => format!("Image {}: ", i + 1),
            TextTask::Describe if images > 1 => format!("Picture {}: ", i + 1),
            TextTask::Describe => String::new(),
        };
        user.push_str(&format!(
            "{label}<|vision_start|><|image_pad|><|vision_end|>"
        ));
        if images > 1 {
            user.push('\n');
        }
    }
    match request.task {
        TextTask::Describe if prompt.is_empty() => {
            ensure!(images > 0, "add an image to describe, or ask a question");
            user.push_str(if images > 1 { DESCRIBE_MANY } else { DESCRIBE });
        }
        TextTask::Describe => user.push_str(prompt),
        TextTask::EnhancePrompt => {
            ensure!(!prompt.is_empty(), "prompt must not be empty");
            user.push_str("Request: ");
            user.push_str(prompt);
        }
    }
    Ok(format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

/// Writes a reply for `request`. `images` are the request's images resized for
/// the vision tower; `visual` supplies their features, as for the text encoder.
pub fn write(
    request: &TextRequest,
    images: &[RgbaImage],
    weights: &Weights,
    device: &Device,
    observer: &mut Observer<'_>,
    mut visual: impl FnMut(usize, VarBuilder, &mut Observer<'_>) -> Result<Visual>,
) -> Result<Written> {
    ensure!(request.max_tokens > 0, "max tokens must be positive");
    ensure!(
        request.temperature.is_finite() && request.temperature >= 0.0,
        "temperature must not be negative"
    );
    ensure!(
        weights.checkpoint().writes_text(),
        "{} has no language-model head; text needs the bf16 or turbo model",
        weights.checkpoint().name()
    );
    let tokenizer = text::tokenizer(weights)?;
    let tokens = text::tokenize(&tokenizer, &chat(request)?, images)?;
    // Generation reads every layer once per token, so keep them on the device.
    let vb = cached_builder(weights.builder("text_encoder", DType::BF16, device)?);
    ensure!(
        vb.contains_tensor("lm_head.weight"),
        "the checkpoint's text encoder has no lm_head.weight"
    );
    let visual = (0..images.len())
        .map(|i| visual(i, vb.pp("model.visual"), observer))
        .collect::<Result<Vec<_>>>()?;
    let language = vb.pp("model.language_model");
    let norm = language.get(4096, "norm.weight")?;
    let head = ops::linear(vb.pp("lm_head"), 4096, VOCABULARY, false)?;
    let logits = |x: &Tensor| -> Result<Vec<f32>> {
        let last = x.narrow(1, x.dim(1)? - 1, 1)?;
        Ok(head
            .forward(&ops::rms(&last, &norm, 1e-6)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?)
    };

    observer.progress(Stage::TextEncoding, 0, text::LAYERS)?;
    let length = tokens.ids.len() + request.max_tokens;
    let mut caches: Vec<_> = (0..text::LAYERS)
        .map(|_| text::Cache::new(2, length))
        .collect();
    let mut x = text::embed(&language, &tokens.ids, device)?;
    let (cos, sin) = text::rotary(&tokens.positions, device)?;
    for (i, cache) in caches.iter_mut().enumerate() {
        x = text::layer(&language, i, &x, &cos, &sin, Some(cache), &tokens.spans, &visual)?;
        device.synchronize()?;
        observer.progress(Stage::TextEncoding, i + 1, text::LAYERS)?;
    }
    drop(visual);

    let mut rng = StdRng::seed_from_u64(request.seed);
    let mut written: Vec<u32> = Vec::new();
    let mut seen = vec![false; VOCABULARY];
    let mut emitted = String::new();
    let mut scores = logits(&x)?;
    let total = request.max_tokens;
    observer.progress(Stage::TextGeneration, 0, total)?;
    while written.len() < total {
        let token = sample(&mut scores, &seen, request.temperature, &mut rng);
        if STOP.contains(&token) {
            break;
        }
        written.push(token);
        seen[token as usize] = true;
        // Decoding the whole reply keeps multi-token characters intact; hold
        // back a trailing partial character until its remaining bytes arrive.
        let text = tokenizer
            .decode(&written, true)
            .map_err(anyhow::Error::msg)?;
        if !text.ends_with('\u{FFFD}')
            && let Some(delta) = text.strip_prefix(emitted.as_str())
            && !delta.is_empty()
        {
            let delta = delta.to_owned();
            emitted.push_str(&delta);
            observer.emit(Event::Text { delta })?;
        }
        observer.progress(Stage::TextGeneration, written.len(), total)?;
        if written.len() == total {
            break;
        }
        let position = tokens.next + written.len() - 1;
        let (cos, sin) = text::rotary(&[[position; 3]], device)?;
        let mut x = text::embed(&language, &[token], device)?;
        for (i, cache) in caches.iter_mut().enumerate() {
            x = text::layer(&language, i, &x, &cos, &sin, Some(cache), &[], &[])?;
        }
        scores = logits(&x)?;
        observer.check()?;
    }
    let text = tokenizer
        .decode(&written, true)
        .map_err(anyhow::Error::msg)?;
    let tokens = written.len();
    Ok(match request.task {
        TextTask::Describe => Written {
            text: text.trim().to_owned(),
            ratio: None,
            tokens,
        },
        TextTask::EnhancePrompt => {
            let (prompt, ratio) = parse_enhanced(&text);
            ensure!(!prompt.is_empty(), "the model returned an empty prompt");
            Written {
                text: prompt,
                ratio: ratio.filter(|_| images.is_empty()),
                tokens,
            }
        }
    })
}

/// Picks the next token from `scores`, which it modifies. Zero temperature
/// picks the most likely token; otherwise samples the top-k/top-p nucleus.
fn sample(scores: &mut [f32], seen: &[bool], temperature: f32, rng: &mut impl Rng) -> u32 {
    for (score, _) in scores.iter_mut().zip(seen).filter(|(_, seen)| **seen) {
        *score -= PRESENCE_PENALTY;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    let k = TOP_K.min(order.len());
    order.select_nth_unstable_by(k - 1, |a, b| scores[*b].total_cmp(&scores[*a]));
    order.truncate(k);
    order.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    if temperature == 0.0 {
        return order[0] as u32;
    }
    let max = scores[order[0]];
    let mut weights: Vec<f32> = order
        .iter()
        .map(|&i| ((scores[i] - max) / temperature).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    let mut cumulative = 0.0;
    let mut keep = weights.len();
    for (n, w) in weights.iter_mut().enumerate() {
        *w /= sum;
        cumulative += *w;
        if cumulative >= TOP_P {
            keep = n + 1;
            break;
        }
    }
    let nucleus: f32 = weights[..keep].iter().sum();
    let mut pick = rng.random::<f32>() * nucleus;
    for (i, w) in order.iter().zip(&weights[..keep]) {
        pick -= w;
        if pick <= 0.0 {
            return *i as u32;
        }
    }
    order[keep - 1] as u32
}

/// The prompt and suggested ratio from the model's JSON reply. A reply that
/// is not the requested JSON is used as the prompt itself.
fn parse_enhanced(reply: &str) -> (String, Option<String>) {
    let json = match (reply.find('{'), reply.rfind('}')) {
        (Some(start), Some(end)) if start < end => &reply[start..=end],
        _ => "",
    };
    let Ok(serde_json::Value::Object(fields)) = serde_json::from_str(json) else {
        let text = reply.trim().trim_matches('`').trim();
        return (text.trim_matches('"').trim().to_owned(), None);
    };
    let prompt = fields
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or_default()
        .trim()
        .to_owned();
    let ratio = fields
        .get("ratio")
        .and_then(|r| r.as_str())
        .map(str::trim)
        .filter(|r| ASPECT_RATIOS.contains(r))
        .map(str::to_owned);
    (prompt, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_replies_with_fences_and_unknown_ratios() {
        assert_eq!(
            parse_enhanced("```json\n{\"prompt\": \" A red fox. \", \"ratio\": \"16:9\"}\n```"),
            ("A red fox.".into(), Some("16:9".into()))
        );
        assert_eq!(
            parse_enhanced("{\"prompt\": \"A fox\", \"ratio\": \"21:9\"}"),
            ("A fox".into(), None)
        );
        assert_eq!(
            parse_enhanced("\"A fox in snow\""),
            ("A fox in snow".into(), None)
        );
    }

    #[test]
    fn greedy_sampling_penalizes_repeated_tokens() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut scores = vec![0.0, 3.0, 2.5, 1.0];
        assert_eq!(sample(&mut scores.clone(), &[false; 4], 0.0, &mut rng), 1);
        assert_eq!(
            sample(&mut scores, &[false, true, false, false], 0.0, &mut rng),
            2
        );
    }

    #[test]
    fn sampling_stays_within_the_nucleus() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let mut scores = vec![10.0, 9.5, -20.0, -30.0];
            let token = sample(&mut scores, &[false; 4], 1.0, &mut rng);
            assert!(token < 2, "sampled {token}");
        }
    }

    #[test]
    fn chats_label_images_for_each_task() -> Result<()> {
        let image = RgbaImage::new(64, 64);
        let mut request = TextRequest::new(TextTask::Describe, "");
        ensure!(chat(&request).is_err());
        request.images = vec![image.clone(), image];
        let describe = chat(&request)?;
        assert!(describe.contains("Picture 2: <|vision_start|>"));
        assert!(describe.ends_with("<|im_start|>assistant\n"));
        request.task = TextTask::EnhancePrompt;
        request.prompt = "make it night".into();
        let edit = chat(&request)?;
        assert!(edit.contains("Image 1: <|vision_start|>"));
        assert!(edit.contains("Request: make it night<|im_end|>"));
        Ok(())
    }
}

