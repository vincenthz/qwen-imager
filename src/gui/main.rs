use gpui::{prelude::*, *};
use gpui_component::{Root, Theme, ThemeMode};
use std::borrow::Cow;

mod timing;
mod window;
mod workspaces;

actions!(img_gen_gui, [Quit]);

struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok(match path {
            "icons/save.svg" => Some(Cow::Borrowed(include_bytes!("assets/save.svg"))),
            "icons/image.svg" => Some(Cow::Borrowed(include_bytes!("assets/image.svg"))),
            "icons/bell.svg" => Some(Cow::Borrowed(include_bytes!("assets/bell.svg"))),
            _ => None,
        })
    }

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(["icons/save.svg", "icons/image.svg", "icons/bell.svg"]
            .into_iter()
            .filter(|asset| asset.starts_with(path))
            .map(Into::into)
            .collect())
    }
}

fn main() {
    Application::new().with_assets(Assets).run(|cx: &mut App| {
        gpui_component::init(cx);
        Theme::change(ThemeMode::Dark, None, cx);
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
        cx.on_action(|_: &Quit, cx| cx.quit());
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        let bounds = Bounds::centered(None, size(px(820.), px(850.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(540.), px(600.))),
                titlebar: Some(TitlebarOptions {
                    title: Some("Qwen Image 2.1".into()),
                    ..Default::default()
                }),
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
