use gpui::{App, Rgba};
use gpui_component::{ActiveTheme as _, ThemeMode};
use serde::{Deserialize, Serialize};

/// Light or dark interface, persisted in the settings and toggled from the title bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Appearance {
    #[default]
    Dark,
    Light,
}

impl Appearance {
    pub fn toggled(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }

    pub fn mode(self) -> ThemeMode {
        match self {
            Self::Dark => ThemeMode::Dark,
            Self::Light => ThemeMode::Light,
        }
    }
}

/// Colors for the custom surfaces drawn outside gpui_component widgets.
#[derive(Clone, Copy)]
pub struct Palette {
    pub background: Rgba,
    pub panel: Rgba,
    pub card: Rgba,
    pub node: Rgba,
    pub node_border: Rgba,
    pub canvas: Rgba,
    pub grid: Rgba,
    pub border: Rgba,
    pub text: Rgba,
    pub strong_text: Rgba,
    pub message: Rgba,
    pub muted: Rgba,
    pub subtle: Rgba,
    pub accent: Rgba,
    pub success: Rgba,
    pub warning: Rgba,
}

const DARK: Palette = Palette {
    background: rgb_const(0x15171b),
    panel: rgb_const(0x1c1f24),
    card: rgb_const(0x22262d),
    node: rgb_const(0x232730),
    node_border: rgb_const(0x414957),
    canvas: rgb_const(0x181b20),
    grid: rgb_const(0x353b47),
    border: rgb_const(0x303640),
    text: rgb_const(0xe4e7ec),
    strong_text: rgb_const(0xc4cad4),
    message: rgb_const(0xb8c2d2),
    muted: rgb_const(0x9da6b5),
    subtle: rgb_const(0x8b93a1),
    accent: rgb_const(0x8aa6ff),
    success: rgb_const(0x75cfb8),
    warning: rgb_const(0xf1ae75),
};

const LIGHT: Palette = Palette {
    background: rgb_const(0xf5f6f8),
    panel: rgb_const(0xffffff),
    card: rgb_const(0xeceef2),
    node: rgb_const(0xffffff),
    node_border: rgb_const(0xc8cdd6),
    canvas: rgb_const(0xeef0f3),
    grid: rgb_const(0xc3c8d1),
    border: rgb_const(0xd6dae1),
    text: rgb_const(0x1d2128),
    strong_text: rgb_const(0x353b46),
    message: rgb_const(0x454d5b),
    muted: rgb_const(0x5c6575),
    subtle: rgb_const(0x6a7281),
    accent: rgb_const(0x4569dc),
    success: rgb_const(0x1f8a70),
    warning: rgb_const(0xb2561a),
};

const fn rgb_const(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 16) & 0xff) as f32 / 255.,
        g: ((hex >> 8) & 0xff) as f32 / 255.,
        b: (hex & 0xff) as f32 / 255.,
        a: 1.,
    }
}

/// The palette matching the active gpui_component theme mode.
pub fn palette(cx: &App) -> Palette {
    if cx.theme().is_dark() { DARK } else { LIGHT }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::rgb;

    #[test]
    fn const_colors_match_gpui_rgb() {
        assert_eq!(DARK.accent, rgb(0x8aa6ff));
        assert_eq!(LIGHT.text, rgb(0x1d2128));
    }

    #[test]
    fn appearance_toggles_both_ways() {
        assert_eq!(Appearance::Dark.toggled(), Appearance::Light);
        assert_eq!(Appearance::Light.toggled(), Appearance::Dark);
    }
}
