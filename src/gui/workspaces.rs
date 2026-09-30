use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use gpui::{Context, Entity, ScrollHandle, Subscription, Window, div, prelude::*, px, rgb};
use gpui_component::{Icon, Selectable, button::Button};
use qwen_imager::{ModelOptions, SharedModel};

use crate::window::{ImageWindow, WorkspaceActivity};

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
}

impl Workspaces {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut workspaces = Self {
            tabs: Vec::new(),
            activity: TabActivity::default(),
            tab_scroll: ScrollHandle::new(),
            model: SharedModel::new(ModelOptions {
                offline: true,
                ..Default::default()
            }),
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

    fn select(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.tabs.len() {
            return;
        }
        self.tabs[self.activity.active]
            .view
            .update(cx, |view, _| view.deactivate());
        // Hidden inputs must not continue receiving keyboard events.
        window.blur();
        self.activity.select(index);
        self.tab_scroll.scroll_to_item(index);
        self.tabs[index]
            .view
            .update(cx, |view, cx| view.activate(window, cx));
        cx.notify();
    }
}

impl Render for Workspaces {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0x15171b))
            .text_color(rgb(0xe4e7ec))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .flex_shrink_0()
                    .border_b_1()
                    .border_color(rgb(0x303640))
                    .child(
                        div()
                            .id("workspace-tabs")
                            .flex()
                            .flex_1()
                            .min_w_0()
                            .items_center()
                            .gap_1()
                            .overflow_x_scroll()
                            .track_scroll(&self.tab_scroll)
                            .children(self.tabs.iter().enumerate().map(|(index, _)| {
                                let unread = self.activity.unread.contains(&index);
                                Button::new(("workspace-tab", index))
                                    .flex_shrink_0()
                                    .label(format!("Workspace {}", index + 1))
                                    .selected(self.activity.active == index)
                                    .when(unread, |tab| {
                                        tab.icon(Icon::default().path("icons/bell.svg"))
                                    })
                                    .tooltip(if unread {
                                        "New preview or generation finished"
                                    } else {
                                        "Switch workspace"
                                    })
                                    .on_click(cx.listener(move |view, _, window, cx| {
                                        view.select(index, window, cx)
                                    }))
                            })),
                    )
                    .child(
                        Button::new("new-workspace")
                            .label("+")
                            .w(px(32.))
                            .flex_shrink_0()
                            .tooltip("New workspace")
                            .on_click(cx.listener(|view, _, window, cx| view.add(window, cx))),
                    ),
            )
            .child(
                div()
                    .id(("workspace-content", self.activity.active))
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .child(self.tabs[self.activity.active].view.clone()),
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
