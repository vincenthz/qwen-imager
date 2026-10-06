//! Per-endpoint Bearer credentials live in Keychain, never in settings.json.
use anyhow::{Context as _, Result};
use gpui::{App, Context, Entity, Task, Window, div, prelude::*, rgb};
use gpui_component::{
    Disableable, Sizable,
    button::Button,
    input::{Input, InputState},
};

use crate::settings::server_url;

fn key(address: &str) -> String {
    format!("ImageForger API token: {}", server_url(address))
}

pub fn read_token(address: &str, cx: &App) -> Task<Result<Option<String>>> {
    let read = cx.read_credentials(&key(address));
    cx.background_executor().spawn(async move {
        let credentials = read
            .await
            .context("Could not read the API token from Keychain")?;
        credentials
            .map(|(_, bytes)| {
                String::from_utf8(bytes).map_err(|_| {
                    anyhow::anyhow!("Saved API token is invalid; replace it in Settings → Compute")
                })
            })
            .transpose()
    })
}

/// Deleting a missing Keychain entry is an error on macOS; check first.
fn clear_token(address: &str, cx: &mut App) -> Task<Result<()>> {
    let key = key(address);
    let read = cx.read_credentials(&key);
    cx.spawn(async move |cx| {
        if read.await?.is_some() {
            cx.update(|cx| cx.delete_credentials(&key)).await?;
        }
        Ok(())
    })
}

pub struct HostCredentials {
    address: String,
    input: Entity<InputState>,
    busy: bool,
    has_token: bool,
    status: String,
    failed: bool,
}

impl HostCredentials {
    pub fn new(address: String, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| {
            let mut input = InputState::new(window, cx).placeholder("API token (without Bearer)");
            input.set_masked(true, window, cx);
            input
        });
        let read = read_token(&address, cx);
        cx.spawn_in(window, async move |view, cx| {
            let result = read.await;
            let _ = view.update_in(cx, |view, window, cx| {
                view.busy = false;
                match result {
                    Ok(token) => {
                        view.has_token = token.is_some();
                        view.status = if view.has_token {
                            "Token stored in Keychain"
                        } else {
                            "No token saved"
                        }
                        .into();
                        view.input.update(cx, |input, cx| {
                            input.set_value(token.unwrap_or_default(), window, cx)
                        });
                    }
                    Err(_) => {
                        view.failed = true;
                        view.status = "Could not read Keychain. Try saving the token again.".into();
                    }
                }
                cx.notify();
            });
        })
        .detach();
        Self {
            address,
            input,
            busy: true,
            has_token: false,
            status: "Reading Keychain…".into(),
            failed: false,
        }
    }

    fn save_and_test(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let token = self.input.read(cx).value().trim().to_string();
        if !token.bytes().all(|byte| byte.is_ascii_graphic()) {
            self.status = "Paste the token only, without Bearer, spaces, or line breaks.".into();
            self.failed = true;
            cx.notify();
            return;
        }
        self.busy = true;
        self.failed = false;
        self.status = "Saving token and testing connection…".into();
        let save = if token.is_empty() {
            clear_token(&self.address, cx)
        } else {
            cx.write_credentials(&key(&self.address), "Bearer", token.as_bytes())
        };
        let url = server_url(&self.address);
        cx.spawn_in(window, async move |view, cx| {
            if save.await.is_err() {
                let _ = view.update(cx, |view, cx| {
                    view.busy = false;
                    view.failed = true;
                    view.status =
                        "Could not save the token in Keychain. Connection was not tested.".into();
                    cx.notify();
                });
                return;
            }
            let has_token = !token.is_empty();
            let result = cx
                .background_executor()
                .spawn(async move {
                    image_forger::remote::test_connection(&url, has_token.then_some(token.as_str()))
                })
                .await;
            let _ = view.update(cx, |view, cx| {
                view.busy = false;
                view.has_token = has_token;
                view.failed = result.is_err();
                view.status = match result {
                    Ok(model) => format!("Connected · {model}"),
                    Err(error) => format!("Credentials saved. {error:#}"),
                };
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    fn clear(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        self.busy = true;
        self.status = "Removing token from Keychain…".into();
        let clear = clear_token(&self.address, cx);
        cx.spawn_in(window, async move |view, cx| {
            let result = clear.await;
            let _ = view.update_in(cx, |view, window, cx| {
                view.busy = false;
                view.failed = result.is_err();
                if result.is_ok() {
                    view.has_token = false;
                    view.input
                        .update(cx, |input, cx| input.set_value("", window, cx));
                    view.status = "Token removed from Keychain".into();
                } else {
                    view.status = "Could not remove the token from Keychain.".into();
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }
}

impl Render for HostCredentials {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_2()
            .w_full()
            .child(Input::new(&self.input).disabled(self.busy))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        Button::new("save-test-host")
                            .small()
                            .label("Save & test")
                            .disabled(self.busy)
                            .on_click(
                                cx.listener(|view, _, window, cx| view.save_and_test(window, cx)),
                            ),
                    )
                    .child(
                        Button::new("clear-host-token")
                            .small()
                            .label("Clear token")
                            .disabled(self.busy || !self.has_token)
                            .on_click(cx.listener(|view, _, window, cx| view.clear(window, cx))),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(if self.failed { 0xf1ae75 } else { 0x9da6b5 }))
                    .child(self.status.clone()),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_scoped_to_the_normalized_endpoint() {
        assert_eq!(key("render.local"), key("http://render.local:6996/"));
        assert_ne!(key("render.local:6996"), key("render.local:6997"));
        assert_ne!(
            key("http://render.local:6996"),
            key("https://render.local:6996")
        );
        assert_ne!(key("https://render.local/a"), key("https://render.local/b"));
    }
}
