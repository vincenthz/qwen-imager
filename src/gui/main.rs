use gpui::{prelude::*, *};
use gpui_component::{Root, Theme, ThemeMode, TitleBar};
use std::borrow::Cow;

mod credentials;
mod errors;
mod models;
mod settings;
mod timing;
mod window;
mod workflow;
mod workspaces;

actions!(img_gen_gui, [Quit]);

struct Assets;

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
            "icons/server.svg" => Some(Cow::Borrowed(include_bytes!("assets/server.svg"))),
            _ => None,
        })
    }

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok([
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
            "icons/server.svg",
        ]
        .into_iter()
        .filter(|asset| asset.starts_with(path))
        .map(Into::into)
        .collect())
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
            cx.set_global(settings);
            Theme::change(ThemeMode::Dark, None, cx);
            // Bound after gpui_component so Cmd-Enter in the prompt generates
            // instead of inserting a newline like Enter.
            cx.bind_keys([
                KeyBinding::new("cmd-q", Quit, None),
                KeyBinding::new(
                    "cmd-enter",
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
