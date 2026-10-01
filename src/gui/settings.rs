use std::path::PathBuf;

use anyhow::Context as _;
use gpui::{Context, Entity, EventEmitter, Global, Window, div, prelude::*, px, rgb};
use gpui_component::{
    Disableable, Selectable,
    button::*,
    input::{Input, InputState},
};
use qwen_imager::Checkpoint;
use serde::{Deserialize, Serialize};

/// Application-wide preferences, persisted between launches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Initial values for new workspaces.
    pub steps: usize,
    pub size: u32,
    /// Where the Save dialog opens. None uses ~/Pictures.
    pub output_directory: Option<PathBuf>,
    /// Decode a preview after every step, rather than on request (plus two checkpoints).
    pub automatic_previews: bool,
    /// Pause sampling while a preview decodes, rather than decoding in the background.
    pub sequential_previews: bool,
    /// Compute the denoiser's attention in BF16 rather than float32.
    pub bf16_attention: bool,
    /// Checkpoint that workspaces load; changes apply as each workspace becomes idle.
    pub model: Checkpoint,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            steps: 20,
            size: 512,
            output_directory: None,
            automatic_previews: true,
            sequential_previews: true,
            bf16_attention: false,
            model: Checkpoint::Original,
        }
    }
}

impl Global for Settings {}

impl Settings {
    fn path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Application Support/QwenImager/settings.json"))
    }

    /// Missing or unreadable settings fall back to the defaults.
    pub fn load() -> Self {
        Self::path()
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<Self>(&bytes).ok())
            .filter(|settings| {
                parse_steps(&settings.steps.to_string()).is_ok()
                    && parse_size(&settings.size.to_string()).is_ok()
            })
            .unwrap_or_default()
    }

    fn save(&self) -> anyhow::Result<()> {
        let path = Self::path().context("HOME is not set")?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn save_directory(&self) -> PathBuf {
        self.output_directory
            .clone()
            .filter(|directory| directory.is_dir())
            .or_else(pictures)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }
}

fn pictures() -> Option<PathBuf> {
    let pictures = PathBuf::from(std::env::var_os("HOME")?).join("Pictures");
    pictures.is_dir().then_some(pictures)
}

pub fn parse_steps(text: &str) -> Result<usize, &'static str> {
    text.parse::<usize>()
        .ok()
        .filter(|&steps| steps > 0)
        .ok_or("Enter a positive whole number of steps.")
}

pub fn parse_size(text: &str) -> Result<u32, &'static str> {
    text.parse::<u32>()
        .ok()
        .filter(|size| (32..=2048).contains(size) && size % 32 == 0)
        .ok_or("Size must be a multiple of 32 between 32 and 2048 pixels (square image).")
}

pub fn digits_input(
    window: &mut Window,
    cx: &mut Context<InputState>,
    value: String,
) -> InputState {
    InputState::new(window, cx)
        .default_value(value)
        .validate(|text, _| text.bytes().all(|c| c.is_ascii_digit()))
}

pub enum SettingsEvent {
    /// The new settings are already applied; this carries the previous ones.
    Saved(Settings),
    Dismissed,
}

pub struct SettingsPanel {
    steps: Entity<InputState>,
    size: Entity<InputState>,
    output_directory: Option<PathBuf>,
    automatic_previews: bool,
    sequential_previews: bool,
    bf16_attention: bool,
    model: Checkpoint,
    choosing_directory: bool,
    error: Option<String>,
}

impl EventEmitter<SettingsEvent> for SettingsPanel {}

impl SettingsPanel {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let settings = cx.global::<Settings>().clone();
        Self {
            steps: cx.new(|cx| digits_input(window, cx, settings.steps.to_string())),
            size: cx.new(|cx| digits_input(window, cx, settings.size.to_string())),
            output_directory: settings.output_directory,
            automatic_previews: settings.automatic_previews,
            sequential_previews: settings.sequential_previews,
            bf16_attention: settings.bf16_attention,
            model: settings.model,
            choosing_directory: false,
            error: None,
        }
    }

    fn choose_directory(&mut self, window: &Window, cx: &mut Context<Self>) {
        if self.choosing_directory {
            return;
        }
        self.choosing_directory = true;
        let answer = rfd::AsyncFileDialog::new()
            .set_parent(window)
            .set_title("Choose the output directory")
            .set_directory(cx.global::<Settings>().save_directory())
            .pick_folder();
        cx.spawn(async move |panel, cx| {
            let folder = answer.await;
            let _ = panel.update(cx, |panel, cx| {
                panel.choosing_directory = false;
                if let Some(folder) = folder {
                    panel.output_directory = Some(folder.path().to_owned());
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        let parsed = parse_steps(&self.steps.read(cx).value())
            .and_then(|steps| Ok((steps, parse_size(&self.size.read(cx).value())?)));
        let (steps, size) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                self.error = Some(error.into());
                cx.notify();
                return;
            }
        };
        let settings = Settings {
            steps,
            size,
            output_directory: self.output_directory.clone(),
            automatic_previews: self.automatic_previews,
            sequential_previews: self.sequential_previews,
            bf16_attention: self.bf16_attention,
            model: self.model,
        };
        if let Err(error) = settings.save() {
            self.error = Some(format!("Could not save settings: {error:#}"));
            cx.notify();
            return;
        }
        let previous = std::mem::replace(cx.global_mut::<Settings>(), settings);
        cx.emit(SettingsEvent::Saved(previous));
    }
}

pub fn model_label(model: Checkpoint) -> &'static str {
    match model {
        Checkpoint::Original => "BF16",
        Checkpoint::Mlx8Bit => "MLX 8-bit",
        Checkpoint::Mlx4Bit => "MLX 4-bit",
    }
}

fn section(title: &'static str) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(div().text_sm().text_color(rgb(0x9da6b5)).child(title))
}

fn hint(text: &'static str) -> impl IntoElement {
    div().text_xs().text_color(rgb(0x9da6b5)).child(text)
}

impl Render for SettingsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let directory = match &self.output_directory {
            Some(directory) => directory.display().to_string(),
            None => "~/Pictures (default)".into(),
        };
        div()
            .w(px(440.))
            .max_w_full()
            .flex()
            .flex_col()
            .gap_4()
            .p_4()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x303640))
            .bg(rgb(0x1c1f24))
            .shadow_lg()
            .child(div().text_lg().child("Settings"))
            .child(
                section("New workspaces")
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child("Steps")
                                    .child(Input::new(&self.steps).w(px(72.))),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child("Size (px)")
                                    .child(Input::new(&self.size).w(px(80.))),
                            ),
                    )
                    .child(hint("Idle workspaces still showing the old defaults also update.")),
            )
            .child(
                section("Output directory")
                    .child(
                        div()
                            .min_w_0()
                            .truncate()
                            .text_sm()
                            .child(directory),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                Button::new("choose-directory")
                                    .label("Choose…")
                                    .disabled(self.choosing_directory)
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.choose_directory(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("default-directory")
                                    .label("Use default")
                                    .disabled(self.output_directory.is_none())
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.output_directory = None;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(hint("The Save dialog opens here.")),
            )
            .child(
                section("Previews")
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                Button::new("previews-auto")
                                    .label("Automatic")
                                    .selected(self.automatic_previews)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.automatic_previews = true;
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("previews-manual")
                                    .label("Manual")
                                    .selected(!self.automatic_previews)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.automatic_previews = false;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(hint(if self.automatic_previews {
                        "Show a preview after every step."
                    } else {
                        "Preview on request, plus automatic previews at step 5 and halfway."
                    }))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                Button::new("previews-sequential")
                                    .label("Sequential")
                                    .selected(self.sequential_previews)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.sequential_previews = true;
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("previews-parallel")
                                    .label("Parallel")
                                    .selected(!self.sequential_previews)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.sequential_previews = false;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(hint(if self.sequential_previews {
                        "Sampling pauses while each preview decodes."
                    } else {
                        "Previews decode in the background while sampling continues, skipping steps when the decoder is busy. The step-5 and halfway previews always pause sampling."
                    })),
            )
            .child(
                section("Attention precision")
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                Button::new("attention-f32")
                                    .label("Float32")
                                    .selected(!self.bf16_attention)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.bf16_attention = false;
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("attention-bf16")
                                    .label("BF16")
                                    .selected(self.bf16_attention)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.bf16_attention = true;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(hint(if self.bf16_attention {
                        "Faster sampling with slightly different pixels. A step that overflows is redone in Float32, which the rest of that generation then uses."
                    } else {
                        "The reference path: the denoiser computes attention in Float32."
                    })),
            )
            .child(
                section("Model")
                    .child(div().flex().gap_2().children(Checkpoint::ALL.map(|model| {
                        Button::new(model.name())
                            .label(model_label(model))
                            .selected(self.model == model)
                            .on_click(cx.listener(move |panel, _, _, cx| {
                                panel.model = model;
                                cx.notify();
                            }))
                    })))
                    .child(hint(match self.model {
                        Checkpoint::Original => "The original BF16 weights, about 32 GB. The reference for image quality.",
                        Checkpoint::Mlx8Bit => "8-bit denoiser and text encoder, about 18 GB. Close to the original, with less memory.",
                        Checkpoint::Mlx4Bit => "4-bit denoiser and text encoder, about 11 GB. The least memory; details can differ from the original.",
                    }))
                    .child(hint("Each workspace switches once idle, and offers a download if the model is missing.")),
            )
            .when_some(self.error.clone(), |panel, error| {
                panel.child(div().text_sm().text_color(rgb(0xff8a8a)).child(error))
            })
            .child(
                div()
                    .flex()
                    .justify_end()
                    .gap_2()
                    .child(
                        Button::new("settings-cancel")
                            .label("Cancel")
                            .on_click(cx.listener(|_, _, _, cx| cx.emit(SettingsEvent::Dismissed))),
                    )
                    .child(
                        Button::new("settings-save")
                            .primary()
                            .label("Save")
                            .on_click(cx.listener(|panel, _, _, cx| panel.save(cx))),
                    ),
            )
            .child(hint("Preview and precision settings apply from the next generation."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_fill_missing_fields_with_defaults() {
        let settings = Settings {
            steps: 30,
            size: 1024,
            output_directory: Some("/tmp/out".into()),
            automatic_previews: false,
            sequential_previews: false,
            bf16_attention: true,
            model: Checkpoint::Mlx4Bit,
        };
        let json = serde_json::to_vec(&settings).unwrap();
        assert!(String::from_utf8_lossy(&json).contains(r#""model":"mlx-4bit""#));
        assert_eq!(serde_json::from_slice::<Settings>(&json).unwrap(), settings);
        let partial: Settings = serde_json::from_str(r#"{"steps": 8}"#).unwrap();
        assert_eq!(
            partial,
            Settings {
                steps: 8,
                ..Settings::default()
            }
        );
    }

    #[test]
    fn steps_and_size_validation() {
        assert_eq!(parse_steps("12"), Ok(12));
        assert!(parse_steps("0").is_err() && parse_steps("").is_err());
        assert_eq!(parse_size("512"), Ok(512));
        assert!(parse_size("500").is_err() && parse_size("4096").is_err());
    }
}
