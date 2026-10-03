use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use gpui::{
    Context, Entity, MouseButton, ScrollHandle, Subscription, Window, div, prelude::*, rgb, rgba,
};
use gpui_component::{
    Disableable, Icon, Selectable, Sizable as _, TitleBar,
    button::{Button, ButtonVariants as _},
    menu::{DropdownMenu as _, PopupMenuItem},
    tab::{Tab, TabBar},
};
use image_forger::{Checkpoint, ModelOptions, SharedModel};

use crate::{
    settings::{Backend, Settings, SettingsEvent, SettingsPanel},
    window::{ImageWindow, WorkspaceActivity},
};

#[derive(Default)]
struct TabActivity {
    active: usize,
    unread: BTreeSet<usize>,
}

impl TabActivity {
    fn select(&mut self, index: usize) {
        self.active = index;
        self.unread.remove(&index);
    }

    fn received(&mut self, index: usize) {
        if index != self.active {
            self.unread.insert(index);
        }
    }
}

struct Workspace {
    view: Entity<ImageWindow>,
    _activity: Subscription,
}

pub struct Workspaces {
    tabs: Vec<Workspace>,
    activity: TabActivity,
    tab_scroll: ScrollHandle,
    model: SharedModel,
    settings: Option<(Entity<SettingsPanel>, Subscription)>,
}

fn shared_model(checkpoint: Checkpoint) -> SharedModel {
    SharedModel::new(ModelOptions {
        offline: true,
        checkpoint,
        ..Default::default()
    })
}

impl Workspaces {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut workspaces = Self {
            tabs: Vec::new(),
            activity: TabActivity::default(),
            tab_scroll: ScrollHandle::new(),
            settings: None,
            model: shared_model(cx.global::<Settings>().model),
        };
        workspaces.add(window, cx);
        workspaces
    }

    fn add(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.tabs.len();
        let generator = Arc::new(Mutex::new(self.model.generator()));
        let view = cx.new(|cx| ImageWindow::new(window, cx, generator));
        let subscription = cx.subscribe(&view, move |workspaces, _, _: &WorkspaceActivity, cx| {
            workspaces.activity.received(index);
            cx.notify();
        });
        self.tabs.push(Workspace {
            view,
            _activity: subscription,
        });
        self.select(index, window, cx);
    }

    fn open_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.settings.is_some() {
            return;
        }
        // The hidden workspace inputs must not receive keyboard events.
        window.blur(cx);
        let panel = cx.new(|cx| SettingsPanel::new(window, cx));
        let subscription = cx.subscribe_in(&panel, window, |workspaces, _, event, window, cx| {
            match event {
                SettingsEvent::Saved(previous) => {
                    workspaces.on_settings_saved(&previous, window, cx);
                }
                SettingsEvent::Dismissed => workspaces.close_settings(window, cx),
            }
        });
        self.settings = Some((panel, subscription));
        cx.notify();
    }

    fn on_settings_saved(
        &mut self,
        previous: &Settings,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let model = cx.global::<Settings>().model;
        if model != previous.model {
            // Running generations keep the previous weights until they finish.
            self.model = shared_model(model);
        }
        for tab in &self.tabs {
            let generator = (model != previous.model)
                .then(|| Arc::new(Mutex::new(self.model.generator())));
            tab.view.update(cx, |view, cx| {
                if let Some(generator) = generator {
                    view.set_generator(generator, window, cx);
                }
                view.apply_defaults(previous, window, cx)
            });
        }
        cx.notify();
    }

    fn close_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.settings.take().is_some() {
            window.blur(cx);
            self.tabs[self.activity.active]
                .view
                .update(cx, |view, cx| view.activate(window, cx));
            cx.notify();
        }
    }

    fn select(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.tabs.len() {
            return;
        }
        self.tabs[self.activity.active]
            .view
            .update(cx, |view, _| view.deactivate());
        // Hidden inputs must not continue receiving keyboard events.
        window.blur(cx);
        self.activity.select(index);
        self.tab_scroll.scroll_to_item(index);
        self.tabs[index]
            .view
            .update(cx, |view, cx| view.activate(window, cx));
        cx.notify();
    }

    fn open_unread(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(&index) = self.activity.unread.iter().next() {
            self.select(index, window, cx);
        }
    }
}

impl Render for Workspaces {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let has_unread = !self.activity.unread.is_empty();
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x15171b))
            .text_color(rgb(0xe4e7ec))
            .child(
                TitleBar::new()
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x9da6b5))
                            .child("ImageForger"),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .pr_2()
                            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                            .child(
                                Button::new("compute-backend")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::default().path("icons/server.svg"))
                                    .tooltip("Compute backend")
                                    .dropdown_menu(move |menu, _, cx| {
                                        let current = cx.global::<Backend>().clone();
                                        let servers = cx.global::<Settings>().servers.clone();
                                        let mut menu = menu.item(
                                            PopupMenuItem::new("Local")
                                                .checked(current == Backend::Local)
                                                .on_click(|_, _, cx| {
                                                    cx.set_global(Backend::Local);
                                                }),
                                        );
                                        for server in servers {
                                            let backend = Backend::Remote {
                                                address: server.clone(),
                                            };
                                            let checked = current == backend;
                                            menu = menu.item(
                                                PopupMenuItem::new(server)
                                                    .checked(checked)
                                                    .on_click(move |_, _, cx| {
                                                        cx.set_global(backend.clone());
                                                    }),
                                            );
                                        }
                                        menu
                                    }),
                            )
                            .child(
                                Button::new("notifications")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::default().path("icons/bell.svg"))
                                    .selected(has_unread)
                                    .disabled(!has_unread)
                                    .tooltip(if has_unread {
                                        "Unread activity"
                                    } else {
                                        "No unread activity"
                                    })
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        view.open_unread(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("settings")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::default().path("icons/settings.svg"))
                                    .selected(self.settings.is_some())
                                    .tooltip("Settings")
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        if view.settings.is_some() {
                                            view.close_settings(window, cx);
                                        } else {
                                            view.open_settings(window, cx);
                                        }
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .child(
                        TabBar::new("workspace-tabs")
                            .w_full()
                            .selected_index(self.activity.active)
                            .track_scroll(&self.tab_scroll)
                            .on_click(cx.listener(|view, index, window, cx| {
                                view.select(*index, window, cx)
                            }))
                            .children(self.tabs.iter().enumerate().map(|(index, _)| {
                                let unread = self.activity.unread.contains(&index);
                                Tab::new()
                                    .label(format!("Workspace {}", index + 1))
                                    .when(unread, |tab| {
                                        tab.icon(Icon::default().path("icons/bell.svg"))
                                    })
                            }))
                            .suffix(
                                Button::new("new-workspace")
                                    .xsmall()
                                    .label("+")
                                    .tooltip("New workspace")
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        view.add(window, cx)
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .relative()
                    .child(self.tabs[self.activity.active].view.clone())
                    .when_some(self.settings.as_ref(), |root, (panel, _)| {
                        root.child(
                            div()
                                .id("settings-backdrop")
                                .absolute()
                                .inset_0()
                                .flex()
                                .items_center()
                                .justify_center()
                                .p_3()
                                .bg(rgba(0x0000_0080))
                                .occlude()
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|view, _, window, cx| {
                                        view.close_settings(window, cx)
                                    }),
                                )
                                .child(
                                    div()
                                        .id("settings-panel")
                                        .on_mouse_down(MouseButton::Left, |_, _, cx| {
                                            cx.stop_propagation()
                                        })
                                        .child(panel.clone()),
                                ),
                        )
                    }),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_background_activity_sets_a_bell_and_selecting_clears_it() {
        let mut tabs = TabActivity::default();
        tabs.received(0);
        assert!(tabs.unread.is_empty());
        tabs.received(1); // A preview in a background workspace.
        tabs.received(1); // Completion keeps a single bell.
        tabs.received(2);
        assert_eq!(tabs.unread, BTreeSet::from([1, 2]));
        tabs.select(1);
        assert_eq!(tabs.unread, BTreeSet::from([2]));
        tabs.received(1);
        assert_eq!(tabs.unread, BTreeSet::from([2]));
        tabs.select(0);
        assert_eq!(tabs.unread, BTreeSet::from([2]));
        tabs.received(1); // Later background updates ring again.
        assert_eq!(tabs.unread, BTreeSet::from([1, 2]));
        tabs.select(2);
        assert_eq!(tabs.unread, BTreeSet::from([1]));
    }
}
