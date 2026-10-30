use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use gpui::{
    AnyView, Context, Entity, MouseButton, ScrollHandle, Subscription, Window, div, prelude::*, rgba,
};
use gpui_component::{
    Disableable, Icon, Sizable as _, TitleBar,
    button::{Button, ButtonVariants as _},
    menu::{DropdownMenu as _, PopupMenuItem},
    tab::{Tab, TabBar},
};
use image_forger::{Checkpoint, ModelOptions, SharedModel};

use crate::{
    models::Models,
    palette::{AccentSelected as _, Appearance, Heading as _, palette},
    settings::{
        Backend, Settings, ModelsPanel, SettingsEvent, SettingsPanel, select_appearance, select_backend,
    },
    window::{ImageWindow, OpenPanel, WorkspaceActivity},
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
    _panels: Subscription,
}

pub struct Workspaces {
    tabs: Vec<Workspace>,
    activity: TabActivity,
    tab_scroll: ScrollHandle,
    model: SharedModel,
    downloads: Entity<Models>,
    _downloads_subscription: Subscription,
    _backend_subscription: Subscription,
    _error_subscription: Subscription,
    /// Settings or Models, shown over the active workspace.
    panel: Option<(AnyView, Subscription)>,
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
        let downloads = cx.new(Models::new);
        let subscription = cx.observe(&downloads, |_, _, cx| cx.notify());
        let mut workspaces = Self {
            tabs: Vec::new(),
            activity: TabActivity::default(),
            tab_scroll: ScrollHandle::new(),
            panel: None,
            downloads,
            _downloads_subscription: subscription,
            _backend_subscription: cx.observe_global::<Backend>(|_, cx| cx.notify()),
            _error_subscription: cx.observe_global_in::<crate::errors::ErrorReports>(window, |_, window, cx| crate::errors::present(window, cx)),
            model: shared_model(cx.global::<Settings>().model),
        };
        workspaces.add(window, cx);
        workspaces
    }

    fn add(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.tabs.len();
        let generator = Arc::new(Mutex::new(self.model.generator()));
        let view = cx.new(|cx| ImageWindow::new(window, cx, generator, self.downloads.clone()));
        let subscription = cx.subscribe(&view, move |workspaces, _, _: &WorkspaceActivity, cx| {
            workspaces.activity.received(index);
            cx.notify();
        });
        let panels = cx.subscribe_in(&view, window, |workspaces, _, panel: &OpenPanel, window, cx| {
            match panel {
                OpenPanel::Settings => workspaces.open_settings(window, cx),
                OpenPanel::Models => workspaces.open_models(window, cx),
            }
        });
        self.tabs.push(Workspace {
            view,
            _activity: subscription,
            _panels: panels,
        });
        self.select(index, window, cx);
    }

    fn open_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.panel.is_some() {
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
                SettingsEvent::Dismissed => workspaces.close_panel(window, cx),
            }
        });
        self.panel = Some((panel.into(), subscription));
        cx.notify();
    }

    fn open_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.panel.is_some() {
            return;
        }
        window.blur(cx);
        let panel = cx.new(|cx| ModelsPanel::new(cx, self.downloads.clone()));
        let subscription = cx.subscribe_in(&panel, window, |workspaces, _, event, window, cx| {
            if let SettingsEvent::Dismissed = event {
                workspaces.close_panel(window, cx)
            }
        });
        self.panel = Some((panel.into(), subscription));
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
                    view.set_generator(generator, cx);
                }
                view.apply_defaults(previous, window, cx)
            });
        }
        cx.notify();
    }

    fn close_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.panel.take().is_some() {
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
            .bg(palette(cx).background)
            .text_color(palette(cx).text)
            .child(
                TitleBar::new()
                    .child(
                        div()
                            .text_sm()
                            .text_color(palette(cx).muted)
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
                                    .label(if matches!(cx.global::<Backend>(), Backend::Local) { "Local" } else { "Remote" })
                                    .tooltip(match cx.global::<Backend>() { Backend::Local => "Compute on this Mac".into(), Backend::Remote { address } => format!("Compute on {address}") })
                                    .dropdown_menu(move |menu, _, cx| {
                                        let current = cx.global::<Backend>().clone();
                                        let servers = cx.global::<Settings>().servers.clone();
                                        let mut menu = menu.item(
                                            PopupMenuItem::new("Local")
                                                .checked(current == Backend::Local)
                                                .on_click(|_, _, cx| {
                                                    select_backend(Backend::Local, cx);
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
                                                        select_backend(backend.clone(), cx);
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
                                    .accent_selected(has_unread, cx)
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
                            .child({
                                let appearance = cx.global::<Settings>().appearance;
                                Button::new("appearance")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::default().path(match appearance {
                                        Appearance::Dark => "icons/sun.svg",
                                        Appearance::Light => "icons/moon.svg",
                                    }))
                                    .tooltip(match appearance {
                                        Appearance::Dark => "Switch to light theme",
                                        Appearance::Light => "Switch to dark theme",
                                    })
                                    .on_click(move |_, _, cx| {
                                        select_appearance(appearance.toggled(), cx)
                                    })
                            })
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
                                let active = index == self.activity.active;
                                let accent = palette(cx).accent;
                                // The label is a child so the active tab can carry an accent
                                // underline; a Tab icon would replace it, so the bell is one too.
                                Tab::new()
                                    .child(
                                        div()
                                            .h_full()
                                            .flex()
                                            .items_center()
                                            .heading()
                                            .border_b_2()
                                            .border_color(if active { accent } else { rgba(0) })
                                            .when(active, |label| label.text_color(accent))
                                            .child(format!("Workspace {}", index + 1)),
                                    )
                                    .when(unread, |tab| {
                                        tab.child(
                                            Icon::default()
                                                .path("icons/bell.svg")
                                                .size_3p5()
                                                .text_color(palette(cx).accent),
                                        )
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
                    .when_some(self.panel.as_ref(), |root, (panel, _)| {
                        root.child(
                            div()
                                .id("panel-backdrop")
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
                                        view.close_panel(window, cx)
                                    }),
                                )
                                .child(
                                    div()
                                        .id("panel")
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
