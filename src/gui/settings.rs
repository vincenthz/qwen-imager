use std::path::PathBuf;

use anyhow::Context as _;
use gpui::{
    App, Context, EventEmitter, Global, SharedString, WeakEntity, Window, div, prelude::*, px,
    rgb,
};
use gpui_component::{
    Disableable,
    button::{Button, ButtonVariants as _},
    input::InputState,
    setting::{
        NumberFieldOptions, SettingField, SettingGroup, SettingItem, SettingPage,
        Settings as SettingsComponent,
    },
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
    /// A setting changed; carries the settings before the change.
    Saved(Settings),
    Dismissed,
}

pub struct SettingsPanel;

impl EventEmitter<SettingsEvent> for SettingsPanel {}

impl SettingsPanel {
    pub fn new(_: &mut Window, _: &mut Context<Self>) -> Self {
        Self
    }
}

/// Apply a settings mutation: persist it, replace the global, and notify the
/// workspaces so they can switch models and refresh their defaults.
fn commit(
    panel: &WeakEntity<SettingsPanel>,
    cx: &mut App,
    mutate: impl FnOnce(&mut Settings),
) {
    let previous = cx.global::<Settings>().clone();
    let mut next = previous.clone();
    mutate(&mut next);
    let _ = next.save();
    cx.set_global(next);
    let _ = panel.update(cx, |_, cx| cx.emit(SettingsEvent::Saved(previous)));
}

fn rounded_size(value: f64) -> u32 {
    ((value / 32.0).round() as i64 * 32).clamp(32, 2048) as u32
}

fn output_directory_field(panel: WeakEntity<SettingsPanel>, cx: &mut App) -> gpui::AnyElement {
    let directory = cx.global::<Settings>().save_directory();
    let is_default = cx.global::<Settings>().output_directory.is_none();
    let path = directory.display().to_string();
    let choose_panel = panel.clone();
    let default_panel = panel;

    div()
        .flex()
        .items_center()
        .gap_2()
        .min_w_0()
        .child(div().min_w_0().truncate().text_sm().child(path))
        .child(
            Button::new("choose-directory")
                .label("Choose…")
                .on_click(move |_, window, cx| {
                    let answer = rfd::AsyncFileDialog::new()
                        .set_parent(window)
                        .set_title("Choose the output directory")
                        .set_directory(cx.global::<Settings>().save_directory())
                        .pick_folder();
                    let panel = choose_panel.clone();
                    cx.spawn(async move |cx| {
                        let folder = answer.await;
                        if let Some(folder) = folder {
                            let path = folder.path().to_owned();
                            let _ = cx.update(|cx| {
                                commit(&panel, cx, |s| s.output_directory = Some(path))
                            });
                        }
                    })
                    .detach();
                }),
        )
        .child(
            Button::new("default-directory")
                .label("Use default")
                .disabled(is_default)
                .on_click(move |_, _, cx| {
                    commit(&default_panel, cx, |s| s.output_directory = None)
                }),
        )
        .into_any_element()
}

impl Render for SettingsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let panel = cx.weak_entity();
        let defaults = Settings::default();
        let model_options: Vec<(SharedString, SharedString)> = Checkpoint::ALL
            .iter()
            .map(|&model| (model.name().into(), model_label(model).into()))
            .collect();

        div()
            .flex()
            .flex_col()
            .w(px(760.))
            .h(px(680.))
            .max_w_full()
            .max_h_full()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0x303640))
            .bg(rgb(0x1c1f24))
            .shadow_lg()
            .overflow_hidden()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_4()
                    .py_3()
                    .border_b_1()
                    .border_color(rgb(0x303640))
                    .child(div().text_lg().child("Settings"))
                    .child(
                        Button::new("settings-done")
                            .primary()
                            .label("Done")
                            .on_click(cx.listener(|_, _, _, cx| {
                                cx.emit(SettingsEvent::Dismissed)
                            })),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .child(
                        SettingsComponent::new("settings")
                            .page(
                                SettingPage::new("General")
                                    .group(
                                        SettingGroup::new()
                                            .title("New workspaces")
                                            .item(
                                                SettingItem::new(
                                                    "Steps",
                                                    SettingField::number_input(
                                                        NumberFieldOptions {
                                                            min: 1.0,
                                                            max: 1000.0,
                                                            step: 1.0,
                                                        },
                                                        |cx| {
                                                            cx.global::<Settings>().steps as f64
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                commit(&panel, cx, |s| {
                                                                    s.steps =
                                                                        (value.round() as usize)
                                                                            .max(1)
                                                                })
                                                            }
                                                        },
                                                    )
                                                    .default_value(defaults.steps as f64),
                                                )
                                                .description(
                                                    "The default denoising step count for new workspaces.",
                                                ),
                                            )
                                            .item(
                                                SettingItem::new(
                                                    "Size (px)",
                                                    SettingField::number_input(
                                                        NumberFieldOptions {
                                                            min: 32.0,
                                                            max: 2048.0,
                                                            step: 32.0,
                                                        },
                                                        |cx| {
                                                            cx.global::<Settings>().size as f64
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                commit(&panel, cx, |s| {
                                                                    s.size = rounded_size(value)
                                                                })
                                                            }
                                                        },
                                                    )
                                                    .default_value(defaults.size as f64),
                                                )
                                                .description(
                                                    "The default square output size, a multiple of 32.",
                                                ),
                                            )
                                    )
                                    .group(
                                        SettingGroup::new()
                                            .title("Output directory")
                                            .item(
                                                SettingItem::new(
                                                    "Output directory",
                                                    SettingField::render({
                                                        let panel = panel.clone();
                                                        move |_, _window, cx| {
                                                            output_directory_field(panel.clone(), cx)
                                                        }
                                                    }),
                                                )
                                                .description("The Save dialog opens here."),
                                            )
                                    )
                            )
                            .page(
                                SettingPage::new("Generation")
                                    .group(
                                        SettingGroup::new()
                                            .title("Previews")
                                            .item(
                                                SettingItem::new(
                                                    "Automatic previews",
                                                    SettingField::switch(
                                                        |cx| {
                                                            cx.global::<Settings>()
                                                                .automatic_previews
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                commit(&panel, cx, |s| {
                                                                    s.automatic_previews = value
                                                                })
                                                            }
                                                        },
                                                    )
                                                    .default_value(defaults.automatic_previews),
                                                )
                                                .description(
                                                    "Decode a preview after every step.",
                                                ),
                                            )
                                            .item(
                                                SettingItem::new(
                                                    "Sequential previews",
                                                    SettingField::switch(
                                                        |cx| {
                                                            cx.global::<Settings>()
                                                                .sequential_previews
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                commit(&panel, cx, |s| {
                                                                    s.sequential_previews = value
                                                                })
                                                            }
                                                        },
                                                    )
                                                    .default_value(defaults.sequential_previews),
                                                )
                                                .description(
                                                    "Pause sampling while each preview decodes.",
                                                ),
                                            )
                                    )
                                    .group(
                                        SettingGroup::new()
                                            .title("Attention precision")
                                            .item(
                                                SettingItem::new(
                                                    "BF16 attention",
                                                    SettingField::switch(
                                                        |cx| {
                                                            cx.global::<Settings>().bf16_attention
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                commit(&panel, cx, |s| {
                                                                    s.bf16_attention = value
                                                                })
                                                            }
                                                        },
                                                    )
                                                    .default_value(defaults.bf16_attention),
                                                )
                                                .description(
                                                    "Faster sampling with slightly different pixels.",
                                                ),
                                            )
                                    )
                                    .group(
                                        SettingGroup::new()
                                            .title("Model")
                                            .item(
                                                SettingItem::new(
                                                    "Checkpoint",
                                                    SettingField::dropdown(
                                                        model_options,
                                                        |cx| {
                                                            cx.global::<Settings>()
                                                                .model
                                                                .name()
                                                                .into()
                                                        },
                                                        {
                                                            let panel = panel.clone();
                                                            move |value, cx| {
                                                                if let Some(model) =
                                                                    Checkpoint::ALL
                                                                        .into_iter()
                                                                        .find(|m| {
                                                                            m.name()
                                                                                == value.as_str()
                                                                        })
                                                                {
                                                                    commit(&panel, cx, |s| {
                                                                        s.model = model
                                                                    });
                                                                }
                                                            }
                                                        },
                                                    )
                                                    .default_value(SharedString::from(
                                                        defaults.model.name(),
                                                    )),
                                                )
                                                .description(
                                                    "The model weights used by new workspaces.",
                                                ),
                                            )
                                    )
                            )
                    )
            )
    }
}

pub fn model_label(model: Checkpoint) -> &'static str {
    match model {
        Checkpoint::Original => "BF16",
        Checkpoint::Mlx8Bit => "MLX 8-bit",
        Checkpoint::Mlx4Bit => "MLX 4-bit",
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

    #[test]
    fn size_rounds_to_nearest_multiple_of_32() {
        assert_eq!(rounded_size(512.0), 512);
        assert_eq!(rounded_size(513.0), 512);
        assert_eq!(rounded_size(527.0), 512);
        assert_eq!(rounded_size(0.0), 32);
        assert_eq!(rounded_size(4096.0), 2048);
    }
}
