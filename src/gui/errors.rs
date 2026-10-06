//! Errors are owned by the window root, so they survive Settings/workspace changes.
use gpui::{App, ClipboardItem, Global, Window, div, prelude::*, px};
use gpui_component::{Sizable, WindowExt, button::Button, notification::Notification};
use image_forger::diagnostics;

#[derive(Clone)]
struct Report {
    operation: String,
    details: String,
    timestamp: u128,
}

#[derive(Default)]
pub struct ErrorReports {
    pending: Vec<Report>,
}
impl Global for ErrorReports {}

pub fn report(operation: impl Into<String>, error: &anyhow::Error, cx: &mut App) {
    let report = Report {
        operation: operation.into(),
        details: diagnostics::redact(&format!("{error:#}"), &[]),
        timestamp: diagnostics::timestamp_ms(),
    };
    diagnostics::log(
        "ERROR",
        "gui.operation_failed",
        serde_json::json!({"operation": report.operation, "causes": report.details}),
    );
    cx.update_global::<ErrorReports, _>(|reports, _| reports.pending.push(report));
}

pub fn present(window: &mut Window, cx: &mut App) {
    if cx.global::<ErrorReports>().pending.is_empty() {
        return;
    }
    let pending =
        cx.update_global::<ErrorReports, _>(|reports, _| std::mem::take(&mut reports.pending));
    for report in pending {
        let summary: String = report.details.chars().take(180).collect();
        window.push_notification(
            Notification::error(summary)
                .id1::<ErrorReports>(report.operation.clone())
                .title(report.operation.clone())
                .autohide(false)
                .action(move |_, _, _| {
                    let report = report.clone();
                    Button::new("error-details")
                        .small()
                        .label("Details")
                        .on_click(move |_, window, cx| {
                            let report = report.clone();
                            window.open_dialog(cx, move |dialog, _, _| {
                                let text = format!(
                                    "Operation: {}\nTimestamp (Unix ms): {}\n\n{}",
                                    report.operation, report.timestamp, report.details
                                );
                                let copy = text.clone();
                                dialog
                                    .title(report.operation.clone())
                                    .w(px(660.))
                                    .child(
                                        div()
                                            .id("error-diagnostics")
                                            .h(px(300.))
                                            .overflow_y_scroll()
                                            .text_sm()
                                            .children(
                                                text.lines()
                                                    .map(|line| div().child(line.to_owned())),
                                            ),
                                    )
                                    .footer(
                                        div()
                                            .flex()
                                            .gap_2()
                                            .child(
                                                Button::new("copy-error")
                                                    .label("Copy details")
                                                    .on_click(move |_, _, cx| {
                                                        cx.write_to_clipboard(
                                                            ClipboardItem::new_string(copy.clone()),
                                                        )
                                                    }),
                                            )
                                            .child(
                                                Button::new("close-error").label("Close").on_click(
                                                    |_, window, cx| window.close_dialog(cx),
                                                ),
                                            ),
                                    )
                            });
                        })
                }),
            cx,
        );
    }
}
