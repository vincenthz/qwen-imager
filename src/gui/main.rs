use gpui::{prelude::*, *};
use gpui_component::{Root, Theme, TitleBar};
use std::borrow::Cow;

mod credentials;
mod errors;
mod models;
mod palette;
mod settings;
mod timing;
mod viewer;
mod window;
mod workflow;
mod workspaces;

actions!(img_gen_gui, [Quit]);

struct Assets;

// gpui_component's own icons, such as the window controls it draws on Linux,
// come from its default set; the application's icons take precedence.
impl AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok(match path {
            "icons/download.svg" => Some(Cow::Borrowed(include_bytes!("assets/download.svg"))),
            "icons/downloaded.svg" => Some(Cow::Borrowed(include_bytes!("assets/downloaded.svg"))),
            "icons/save.svg" => Some(Cow::Borrowed(include_bytes!("assets/save.svg"))),
            "icons/image.svg" => Some(Cow::Borrowed(include_bytes!("assets/image.svg"))),
            "icons/bell.svg" => Some(Cow::Borrowed(include_bytes!("assets/bell.svg"))),
            "icons/settings.svg" => Some(Cow::Borrowed(include_bytes!("assets/settings.svg"))),
            "icons/pause.svg" => Some(Cow::Borrowed(include_bytes!("assets/pause.svg"))),
            "icons/play.svg" => Some(Cow::Borrowed(include_bytes!("assets/play.svg"))),
            "icons/sparkles.svg" => Some(Cow::Borrowed(include_bytes!("assets/sparkles.svg"))),
            "icons/sliders.svg" => Some(Cow::Borrowed(include_bytes!("assets/sliders.svg"))),
            "icons/workflow.svg" => Some(Cow::Borrowed(include_bytes!("assets/workflow.svg"))),
            "icons/trash.svg" => Some(Cow::Borrowed(include_bytes!("assets/trash.svg"))),
            "icons/star.svg" => Some(Cow::Borrowed(include_bytes!("assets/star.svg"))),
            "icons/server.svg" => Some(Cow::Borrowed(include_bytes!("assets/server.svg"))),
            "icons/sun.svg" => Some(Cow::Borrowed(include_bytes!("assets/sun.svg"))),
            "icons/moon.svg" => Some(Cow::Borrowed(include_bytes!("assets/moon.svg"))),
            "icons/use-input.svg" => Some(Cow::Borrowed(include_bytes!("assets/use-input.svg"))),
            "icons/panel-close.svg" => Some(Cow::Borrowed(include_bytes!("assets/panel-close.svg"))),
            "icons/panel-open.svg" => Some(Cow::Borrowed(include_bytes!("assets/panel-open.svg"))),
            // Used by gpui_component menus and dropdown buttons.
            "icons/check.svg" => Some(Cow::Borrowed(include_bytes!("assets/check.svg"))),
            "icons/chevron-down.svg" => Some(Cow::Borrowed(include_bytes!("assets/chevron-down.svg"))),
            _ => return gpui_kit_assets::Assets.load(path),
        })
    }

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>> {
        let mut assets = [
            "icons/download.svg",
            "icons/downloaded.svg",
            "icons/save.svg",
            "icons/image.svg",
            "icons/bell.svg",
            "icons/settings.svg",
            "icons/pause.svg",
            "icons/play.svg",
            "icons/sparkles.svg",
            "icons/sliders.svg",
            "icons/workflow.svg",
            "icons/trash.svg",
            "icons/star.svg",
            "icons/server.svg",
            "icons/sun.svg",
            "icons/moon.svg",
            "icons/use-input.svg",
            "icons/panel-close.svg",
            "icons/panel-open.svg",
            "icons/check.svg",
            "icons/chevron-down.svg",
        ]
        .into_iter()
        .filter(|asset| asset.starts_with(path))
        .map(SharedString::from)
        .collect::<Vec<_>>();
        assets.extend(gpui_kit_assets::Assets.list(path)?);
        assets.sort();
        assets.dedup();
        Ok(assets)
    }
}

fn main() {
    gpui_platform::application()
        .with_assets(Assets)
        .run(|cx: &mut App| {
            gpui_component::init(cx);
            cx.set_global(errors::ErrorReports::default());
            let settings = settings::Settings::load();
            cx.set_global(settings.backend.clone());
            let appearance = settings.appearance;
            cx.set_global(settings);
            Theme::change(appearance.mode(), None, cx);
            // Bound after gpui_component so Cmd-Enter (Ctrl-Enter off macOS)
            // in the prompt generates instead of inserting a newline like Enter.
            cx.bind_keys([
                KeyBinding::new("secondary-q", Quit, None),
                KeyBinding::new(
                    "secondary-enter",
                    window::Generate,
                    Some(&format!("{} > Input", window::PROMPT_CONTEXT)),
                ),
            ]);
            cx.on_action(|_: &Quit, cx| cx.quit());
            cx.on_window_closed(|cx, _| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();
            let bounds = Bounds::centered(None, size(px(1280.), px(850.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(900.), px(600.))),
                    titlebar: Some(TitlebarOptions {
                        title: Some("ImageForger".into()),
                        ..TitleBar::title_bar_options()
                    }),
                    app_owns_titlebar_drag: true,
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|cx| workspaces::Workspaces::new(window, cx));
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("opening the image generation window");
            cx.activate(true);
        });
}

#[cfg(test)]
mod tests {
    use super::Assets;
    use gpui::AssetSource as _;

    #[test]
    fn component_icons_fall_back_to_the_default_set() -> anyhow::Result<()> {
        for icon in ["window-close", "window-minimize", "window-maximize", "window-restore"] {
            let path = format!("icons/{icon}.svg");
            assert!(Assets.load(&path)?.is_some(), "{path}");
            assert!(Assets.list("icons/")?.contains(&path.into()));
        }
        let ours: &[u8] = include_bytes!("assets/check.svg");
        assert_eq!(Assets.load("icons/check.svg")?.as_deref(), Some(ours));
        Ok(())
    }
}
