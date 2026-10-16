//! Full-size image viewer opened by double-clicking an image.
use std::{cell::Cell, rc::Rc, sync::Arc};

use gpui::{
    Bounds, Context, Corners, CursorStyle, FocusHandle, MouseButton, MouseDownEvent,
    MouseMoveEvent, Pixels, Point, RenderImage, Size, Window, canvas, div, point, prelude::*,
    px, size,
};
use gpui_component::{Disableable, Selectable, Sizable as _, button::Button};

use crate::palette::palette;

/// Zoom steps of the buttons and keyboard shortcuts.
const ZOOM_STEP: f32 = 1.25;
/// The largest magnification, in screen pixels per image pixel.
const MAX_SCALE: f32 = 16.;

pub enum ViewerEvent {
    Close,
}

/// Zoom and pan of an image inside the viewer. A zoom of 1 fits the whole image.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Viewport {
    zoom: f32,
    // Offset of the image centre from the view centre.
    offset: Point<Pixels>,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            zoom: 1.,
            offset: Point::default(),
        }
    }
}

impl Viewport {
    /// Screen pixels per image pixel when the whole image fits the view.
    fn fit_scale(view: Size<Pixels>, (width, height): (f32, f32)) -> f32 {
        (f32::from(view.width) / width).min(f32::from(view.height) / height)
    }

    fn max_zoom(view: Size<Pixels>, image: (f32, f32)) -> f32 {
        (MAX_SCALE / Self::fit_scale(view, image)).max(1.)
    }

    fn scale(&self, view: Size<Pixels>, image: (f32, f32)) -> f32 {
        Self::fit_scale(view, image) * self.zoom
    }

    /// Keeps the image centred on axes where it fits, and its edges inside the
    /// view on axes where it does not.
    fn clamped(self, view: Size<Pixels>, image: (f32, f32)) -> Self {
        let scale = self.scale(view, image);
        let limit = |extent: f32, view: Pixels| ((extent * scale - f32::from(view)) / 2.).max(0.);
        let (x, y) = (limit(image.0, view.width), limit(image.1, view.height));
        Self {
            zoom: self.zoom,
            offset: point(
                px(f32::from(self.offset.x).clamp(-x, x)),
                px(f32::from(self.offset.y).clamp(-y, y)),
            ),
        }
    }

    /// Where the image is drawn in window coordinates.
    fn image_bounds(&self, view: Bounds<Pixels>, image: (f32, f32)) -> Bounds<Pixels> {
        let scale = self.scale(view.size, image);
        let shown = size(px(image.0 * scale), px(image.1 * scale));
        let center = view.center() + self.clamped(view.size, image).offset;
        Bounds::new(
            point(center.x - shown.width / 2., center.y - shown.height / 2.),
            shown,
        )
    }

    /// Zooms to `zoom`, keeping the image point under `anchor` (in window
    /// coordinates) in place.
    fn zoom_to(&mut self, zoom: f32, anchor: Point<Pixels>, view: Bounds<Pixels>, image: (f32, f32)) {
        let zoom = zoom.clamp(1., Self::max_zoom(view.size, image));
        let ratio = zoom / self.zoom;
        let from_center = anchor - view.center();
        let offset = self.clamped(view.size, image).offset;
        self.offset = from_center - (from_center - offset) * ratio;
        self.zoom = zoom;
        *self = self.clamped(view.size, image);
    }

    fn pan(&mut self, delta: Point<Pixels>, view: Size<Pixels>, image: (f32, f32)) {
        self.offset = self.clamped(view, image).offset + delta;
        *self = self.clamped(view, image);
    }
}

pub struct ImageViewer {
    image: Arc<RenderImage>,
    viewport: Viewport,
    // Last pointer position of a drag that pans the image.
    drag: Option<Point<Pixels>>,
    // Window-space bounds of the image area, recorded at paint time.
    bounds: Rc<Cell<Bounds<Pixels>>>,
    focus: FocusHandle,
}

impl gpui::EventEmitter<ViewerEvent> for ImageViewer {}

impl ImageViewer {
    /// The viewer owns `image`; its texture is dropped when the viewer closes.
    pub fn new(image: Arc<RenderImage>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        Self {
            image,
            viewport: Viewport::default(),
            drag: None,
            bounds: Rc::default(),
            focus,
        }
    }

    pub fn image(&self) -> &Arc<RenderImage> {
        &self.image
    }

    fn dimensions(&self) -> (f32, f32) {
        let size = self.image.size(0);
        (size.width.0.max(1) as f32, size.height.0.max(1) as f32)
    }

    fn view(&self) -> Option<Bounds<Pixels>> {
        let view = self.bounds.get();
        (view.size.width > px(0.) && view.size.height > px(0.)).then_some(view)
    }

    fn zoom_at(&mut self, zoom: f32, anchor: Option<Point<Pixels>>, cx: &mut Context<Self>) {
        let image = self.dimensions();
        if let Some(view) = self.view() {
            self.viewport.zoom_to(zoom, anchor.unwrap_or(view.center()), view, image);
            cx.notify();
        }
    }

    fn zoom_by(&mut self, factor: f32, anchor: Option<Point<Pixels>>, cx: &mut Context<Self>) {
        self.zoom_at(self.viewport.zoom * factor, anchor, cx);
    }

    /// The zoom at which one image pixel covers one screen pixel.
    fn actual_size_zoom(&self) -> Option<f32> {
        self.view()
            .map(|view| 1. / Viewport::fit_scale(view.size, self.dimensions()))
    }

    fn fit(&mut self, cx: &mut Context<Self>) {
        self.viewport = Viewport::default();
        cx.notify();
    }

    /// Double-click toggles between fitting the view and actual pixels (or 2× for
    /// images smaller than the view).
    fn toggle_zoom(&mut self, anchor: Point<Pixels>, cx: &mut Context<Self>) {
        if self.viewport.zoom > 1.001 {
            self.fit(cx);
        } else if let Some(actual) = self.actual_size_zoom() {
            self.zoom_at(if actual > 1.1 { actual } else { 2. }, Some(anchor), cx);
        }
    }

    fn start_drag(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        self.drag = Some(event.position);
        cx.notify();
    }

    fn extend_drag(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        let Some(last) = self.drag else {
            return;
        };
        if event.pressed_button != Some(MouseButton::Left) {
            self.drag = None;
            cx.notify();
            return;
        }
        if let Some(view) = self.view() {
            self.viewport.pan(event.position - last, view.size, self.dimensions());
        }
        self.drag = Some(event.position);
        cx.notify();
    }

    fn close(&mut self, cx: &mut Context<Self>) {
        cx.emit(ViewerEvent::Close);
    }
}

impl Render for ImageViewer {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = palette(cx);
        let image = self.image.clone();
        let dimensions = self.dimensions();
        let viewport = self.viewport;
        let record = self.bounds.clone();
        let zoomed = viewport.zoom > 1.001;
        let scale = self.view().map(|view| viewport.scale(view.size, dimensions));
        let max_zoom = self
            .view()
            .map_or(1., |view| Viewport::max_zoom(view.size, dimensions));
        div()
            .id("image-viewer")
            .track_focus(&self.focus)
            .absolute()
            .inset_0()
            .flex()
            .flex_col()
            .bg(palette.background)
            .occlude()
            .on_key_down(cx.listener(|view, event: &gpui::KeyDownEvent, _, cx| {
                match event.keystroke.key.as_str() {
                    "escape" => view.close(cx),
                    "+" | "=" => view.zoom_by(ZOOM_STEP, None, cx),
                    "-" => view.zoom_by(1. / ZOOM_STEP, None, cx),
                    "0" => view.fit(cx),
                    "1" => {
                        if let Some(actual) = view.actual_size_zoom() {
                            view.zoom_at(actual, None, cx);
                        }
                    }
                    _ => return,
                }
                cx.stop_propagation();
            }))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(palette.border)
                    .bg(palette.panel)
                    .text_sm()
                    .child(
                        Button::new("viewer-zoom-out")
                            .small()
                            .label("−")
                            .tooltip("Zoom out (−)")
                            .disabled(!zoomed)
                            .on_click(cx.listener(|view, _, _, cx| view.zoom_by(1. / ZOOM_STEP, None, cx))),
                    )
                    .child(
                        div()
                            .w(px(52.))
                            .text_center()
                            .child(scale.map_or_else(String::new, |scale| format!("{:.0}%", scale * 100.))),
                    )
                    .child(
                        Button::new("viewer-zoom-in")
                            .small()
                            .label("+")
                            .tooltip("Zoom in (+)")
                            .disabled(viewport.zoom >= max_zoom)
                            .on_click(cx.listener(|view, _, _, cx| view.zoom_by(ZOOM_STEP, None, cx))),
                    )
                    .child(
                        Button::new("viewer-fit")
                            .small()
                            .label("Fit")
                            .tooltip("Fit the whole image (0)")
                            .selected(!zoomed)
                            .on_click(cx.listener(|view, _, _, cx| view.fit(cx))),
                    )
                    .child(
                        Button::new("viewer-actual")
                            .small()
                            .label("1:1")
                            .tooltip("Actual pixels (1)")
                            .on_click(cx.listener(|view, _, _, cx| {
                                if let Some(actual) = view.actual_size_zoom() {
                                    view.zoom_at(actual, None, cx);
                                }
                            })),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(palette.muted)
                            .child(format!(
                                "{}×{} · Scroll or drag to pan · ⌘-scroll or pinch to zoom · Double-click for 1:1",
                                dimensions.0, dimensions.1
                            )),
                    )
                    .child(
                        Button::new("viewer-close")
                            .small()
                            .ml_auto()
                            .label("Close")
                            .tooltip("Close the viewer (Esc)")
                            .on_click(cx.listener(|view, _, _, cx| view.close(cx))),
                    ),
            )
            .child(
                div()
                    .id("viewer-canvas")
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .overflow_hidden()
                    .bg(palette.canvas)
                    .cursor(if self.drag.is_some() && zoomed {
                        CursorStyle::ClosedHand
                    } else if zoomed {
                        CursorStyle::OpenHand
                    } else {
                        CursorStyle::Arrow
                    })
                    .on_mouse_down(MouseButton::Left, cx.listener(|view, event, _, cx| view.start_drag(event, cx)))
                    .on_mouse_move(cx.listener(|view, event, _, cx| view.extend_drag(event, cx)))
                    .on_mouse_up(MouseButton::Left, cx.listener(|view, _, _, cx| {
                        view.drag = None;
                        cx.notify();
                    }))
                    .on_mouse_up_out(MouseButton::Left, cx.listener(|view, _, _, cx| {
                        view.drag = None;
                        cx.notify();
                    }))
                    .on_click(cx.listener(|view, event: &gpui::ClickEvent, _, cx| {
                        if event.click_count() == 2
                            && let Some(position) = event.mouse_position()
                        {
                            view.toggle_zoom(position, cx);
                        }
                    }))
                    .on_scroll_wheel(cx.listener(|view, event: &gpui::ScrollWheelEvent, _, cx| {
                        match event.delta {
                            // Mouse wheels zoom; trackpads pan unless ⌘ or Ctrl is held.
                            gpui::ScrollDelta::Lines(lines) => {
                                view.zoom_by(ZOOM_STEP.powf(lines.y.clamp(-3., 3.)), Some(event.position), cx);
                            }
                            gpui::ScrollDelta::Pixels(delta) => {
                                if event.modifiers.control || event.modifiers.platform {
                                    view.zoom_by((f32::from(delta.y) * 0.005).exp(), Some(event.position), cx);
                                } else if let Some(bounds) = view.view() {
                                    view.viewport.pan(delta, bounds.size, view.dimensions());
                                    cx.notify();
                                }
                            }
                        }
                        cx.stop_propagation();
                    }))
                    .on_pinch(cx.listener(|view, event: &gpui::PinchEvent, _, cx| {
                        view.zoom_by(1. + event.delta, Some(event.position), cx);
                    }))
                    .child(
                        canvas(
                            move |bounds, _, _| record.set(bounds),
                            move |bounds, (), window, _| {
                                let shown = viewport.image_bounds(bounds, dimensions);
                                let _ = window.paint_image(bounds, shown, Corners::default(), image.clone(), 0, false);
                            },
                        )
                        .absolute()
                        .size_full(),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> Bounds<Pixels> {
        Bounds::new(point(px(100.), px(50.)), size(px(400.), px(200.)))
    }

    #[test]
    fn fitting_centres_the_whole_image() {
        let image = (1000., 1000.);
        let shown = Viewport::default().image_bounds(view(), image);
        assert_eq!(shown.size, size(px(200.), px(200.)));
        assert_eq!(shown.center(), view().center());
    }

    #[test]
    fn zooming_keeps_the_anchor_in_place() {
        let image = (1000., 1000.);
        let anchor = point(px(320.), px(110.));
        let mut viewport = Viewport::default();
        let before = viewport.image_bounds(view(), image);
        let fraction = |bounds: Bounds<Pixels>| {
            (
                f32::from(anchor.x - bounds.origin.x) / f32::from(bounds.size.width),
                f32::from(anchor.y - bounds.origin.y) / f32::from(bounds.size.height),
            )
        };
        viewport.zoom_to(3., anchor, view(), image);
        let after = viewport.image_bounds(view(), image);
        let (a, b) = (fraction(before), fraction(after));
        assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4, "{a:?} != {b:?}");
    }

    #[test]
    fn panning_stops_at_the_image_edges() {
        let image = (1000., 1000.);
        let mut viewport = Viewport::default();
        // Fitted: the image is centred and cannot move.
        viewport.pan(point(px(50.), px(50.)), view().size, image);
        assert_eq!(viewport.offset, Point::default());
        // At 4× the image is 800px square in a 400×200 view.
        viewport.zoom_to(4., view().center(), view(), image);
        viewport.pan(point(px(1000.), px(-1000.)), view().size, image);
        assert_eq!(viewport.offset, point(px(200.), px(-300.)));
    }

    #[test]
    fn zoom_stays_between_fit_and_the_maximum() {
        let image = (1000., 1000.);
        let mut viewport = Viewport::default();
        viewport.zoom_to(0.1, view().center(), view(), image);
        assert_eq!(viewport.zoom, 1.);
        viewport.zoom_to(1000., view().center(), view(), image);
        assert_eq!(viewport.zoom, MAX_SCALE / 0.2);
    }
}
