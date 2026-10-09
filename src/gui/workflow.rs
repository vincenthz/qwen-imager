//! Session-local workflow graph and its unbounded, screen-independent canvas.
//! Graph execution is deliberately separate from the existing single-image runner.
use std::{cell::Cell, collections::HashSet, rc::Rc};

use crate::palette::{Palette, palette};
use gpui::{
    Bounds, Context, CursorStyle, FocusHandle, MouseButton, MouseDownEvent, MouseMoveEvent,
    PathBuilder, Pixels, Point, Window, canvas, div, point, prelude::*, px, rgb, size,
};
use gpui_component::{Disableable, Sizable, button::Button};

const WIDTH: f32 = 216.;
const HEIGHT: f32 = 144.;
const MIN_ZOOM: f32 = 0.25;
const MAX_ZOOM: f32 = 2.;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Prompt,
    Image,
    Generate,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DataType {
    Text,
    Image,
}

impl DataType {
    fn color(self) -> u32 {
        match self {
            Self::Text => 0xb59aff,
            Self::Image => 0x75cfb8,
        }
    }
}

impl Kind {
    const ALL: [Self; 4] = [Self::Prompt, Self::Image, Self::Generate, Self::Output];

    fn label(self) -> &'static str {
        match self {
            Self::Prompt => "Prompt",
            Self::Image => "Image",
            Self::Generate => "Generate",
            Self::Output => "Output",
        }
    }

    fn caption(self) -> &'static str {
        match self {
            Self::Prompt => "Text input",
            Self::Image => "Reference input",
            Self::Generate => "Image generation step",
            Self::Output => "Final image",
        }
    }

    fn inputs(self) -> &'static [(DataType, &'static str)] {
        match self {
            Self::Generate => &[(DataType::Text, "Prompt"), (DataType::Image, "Images")],
            Self::Output => &[(DataType::Image, "Image")],
            _ => &[],
        }
    }

    fn output(self) -> Option<DataType> {
        match self {
            Self::Prompt => Some(DataType::Text),
            Self::Image | Self::Generate => Some(DataType::Image),
            Self::Output => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    id: usize,
    kind: Kind,
    position: Point<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge {
    source: usize,
    target: usize,
    input: usize,
}

#[derive(Clone, Copy)]
struct Port {
    node: usize,
    // None denotes an output socket.
    input: Option<usize>,
}

#[derive(Default)]
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    next_id: usize,
}

impl Graph {
    fn add(&mut self, kind: Kind, position: Point<f32>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(Node { id, kind, position });
        id
    }

    fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    fn remove(&mut self, id: usize) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.source != id && e.target != id);
    }

    fn connect(&mut self, a: Port, b: Port) -> Result<(), &'static str> {
        let (source, target, input) = match (a.input, b.input) {
            (None, Some(input)) => (a.node, b.node, input),
            (Some(input), None) => (b.node, a.node, input),
            _ => return Err("Connect an output dot to an input dot."),
        };
        if source == target {
            return Err("A node cannot connect to itself.");
        }
        let source_type = self.node(source).and_then(|n| n.kind.output());
        let target_type = self
            .node(target)
            .and_then(|n| n.kind.inputs().get(input))
            .map(|p| p.0);
        if source_type.is_none() || source_type != target_type {
            return Err("Match the port colors: text to text, image to image.");
        }
        let edge = Edge {
            source,
            target,
            input,
        };
        if self.edges.contains(&edge) {
            return Err("These ports are already connected.");
        }
        let mut pending = vec![target];
        let mut visited = HashSet::new();
        while let Some(id) = pending.pop() {
            if id == source {
                return Err("This connection would create a loop.");
            }
            if visited.insert(id) {
                pending.extend(
                    self.edges
                        .iter()
                        .filter(|e| e.source == id)
                        .map(|e| e.target),
                );
            }
        }
        // Image references are ordered by connection time. Other inputs accept one source.
        let multiple = self
            .node(target)
            .is_some_and(|n| n.kind == Kind::Generate && input == 1);
        if multiple {
            if self
                .edges
                .iter()
                .filter(|e| e.target == target && e.input == input)
                .count()
                >= 10
            {
                return Err("A Generate node accepts up to 10 reference images.");
            }
        } else {
            self.edges
                .retain(|e| e.target != target || e.input != input);
        }
        self.edges.push(edge);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Viewport {
    pan: Point<f32>,
    zoom: f32,
}

impl Viewport {
    fn screen(self, world: Point<f32>) -> Point<f32> {
        point(
            world.x * self.zoom + self.pan.x,
            world.y * self.zoom + self.pan.y,
        )
    }

    fn world(self, screen: Point<f32>) -> Point<f32> {
        point(
            (screen.x - self.pan.x) / self.zoom,
            (screen.y - self.pan.y) / self.zoom,
        )
    }

    fn zoom_at(&mut self, anchor: Point<f32>, zoom: f32) {
        let world = self.world(anchor);
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = point(
            anchor.x - world.x * self.zoom,
            anchor.y - world.y * self.zoom,
        );
    }
}

#[derive(Clone, Copy)]
enum Drag {
    Pan {
        pointer: Point<f32>,
        origin: Point<f32>,
    },
    Node {
        id: usize,
        pointer: Point<f32>,
        origin: Point<f32>,
    },
    Wire {
        port: Port,
        pointer: Point<f32>,
    },
}

pub struct WorkflowCanvas {
    graph: Graph,
    viewport: Viewport,
    bounds: Rc<Cell<Bounds<Pixels>>>,
    focus: FocusHandle,
    selected: Option<usize>,
    drag: Option<Drag>,
    message: String,
}

impl WorkflowCanvas {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut graph = Graph::default();
        let prompt = graph.add(Kind::Prompt, point(0., 0.));
        let generate = graph.add(Kind::Generate, point(300., 80.));
        let output = graph.add(Kind::Output, point(600., 80.));
        graph.edges = vec![
            Edge {
                source: prompt,
                target: generate,
                input: 0,
            },
            Edge {
                source: generate,
                target: output,
                input: 0,
            },
        ];
        Self {
            graph,
            viewport: Viewport {
                pan: point(48., 80.),
                zoom: 0.8,
            },
            bounds: Rc::default(),
            focus: cx.focus_handle(),
            selected: None,
            drag: None,
            message: "Drag between matching dots to connect nodes.".into(),
        }
    }

    fn local(&self, position: Point<Pixels>) -> Point<f32> {
        let origin = self.bounds.get().origin;
        point(
            f32::from(position.x - origin.x),
            f32::from(position.y - origin.y),
        )
    }

    fn port_position(node: &Node, input: Option<usize>) -> Point<f32> {
        point(
            node.position.x + if input.is_some() { 0. } else { WIDTH },
            node.position.y + 78. + input.unwrap_or(0) as f32 * 28.,
        )
    }

    fn port_at(&self, screen: Point<f32>) -> Option<Port> {
        for node in self.graph.nodes.iter().rev() {
            let inputs = (0..node.kind.inputs().len()).map(Some);
            let output = node.kind.output().map(|_| None);
            for input in inputs.chain(output) {
                let p = self.viewport.screen(Self::port_position(node, input));
                if (p.x - screen.x).hypot(p.y - screen.y) <= (10. * self.viewport.zoom).max(8.) {
                    return Some(Port {
                        node: node.id,
                        input,
                    });
                }
            }
        }
        None
    }

    fn begin(&mut self, event: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        self.focus.focus(window, cx);
        let p = self.local(event.position);
        if event.button == MouseButton::Left {
            if let Some(port) = self.port_at(p) {
                self.drag = Some(Drag::Wire { port, pointer: p });
                self.message = "Release on a matching dot. Escape cancels.".into();
                cx.notify();
                return;
            }
            let world = self.viewport.world(p);
            if let Some(node) = self.graph.nodes.iter().rev().find(|n| {
                world.x >= n.position.x
                    && world.x <= n.position.x + WIDTH
                    && world.y >= n.position.y
                    && world.y <= n.position.y + HEIGHT
            }) {
                let id = node.id;
                self.selected = Some(id);
                self.drag = Some(Drag::Node {
                    id,
                    pointer: world,
                    origin: node.position,
                });
                // Raise the selected card, keeping hit-testing and rendering order identical.
                let index = self.graph.nodes.iter().position(|n| n.id == id).unwrap();
                let node = self.graph.nodes.remove(index);
                self.graph.nodes.push(node);
                cx.notify();
                return;
            }
        }
        self.selected = None;
        self.drag = Some(Drag::Pan {
            pointer: p,
            origin: self.viewport.pan,
        });
        cx.notify();
    }

    fn motion(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        // A release outside the window must never leave a sticky drag.
        if event.pressed_button.is_none() {
            if self.drag.take().is_some() {
                cx.notify();
            }
            return;
        }
        let p = self.local(event.position);
        match self.drag {
            Some(Drag::Pan { pointer, origin }) => {
                self.viewport.pan = point(origin.x + p.x - pointer.x, origin.y + p.y - pointer.y);
            }
            Some(Drag::Node {
                id,
                pointer,
                origin,
            }) => {
                let world = self.viewport.world(p);
                if let Some(node) = self.graph.nodes.iter_mut().find(|n| n.id == id) {
                    node.position = point(
                        origin.x + world.x - pointer.x,
                        origin.y + world.y - pointer.y,
                    );
                }
            }
            Some(Drag::Wire { port, .. }) => self.drag = Some(Drag::Wire { port, pointer: p }),
            None => return,
        }
        cx.notify();
    }

    fn finish(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        if let Some(Drag::Wire { port, .. }) = self.drag.take() {
            self.message = if let Some(target) = self.port_at(self.local(position)) {
                match self.graph.connect(port, target) {
                    Ok(()) => "Connected. Select a node to disconnect or remove it.".into(),
                    Err(message) => message.into(),
                }
            } else {
                "Connection cancelled. Release on an input or output dot.".into()
            };
        }
        cx.notify();
    }

    fn add(&mut self, kind: Kind, cx: &mut Context<Self>) {
        let bounds = self.bounds.get();
        let center = self.viewport.world(point(
            f32::from(bounds.size.width) / 2.,
            f32::from(bounds.size.height) / 2.,
        ));
        let mut position = point(center.x - WIDTH / 2., center.y - HEIGHT / 2.);
        while self.graph.nodes.iter().any(|n| {
            (n.position.x - position.x).abs() < 24. && (n.position.y - position.y).abs() < 24.
        }) {
            position.x += 32.;
            position.y += 32.;
        }
        self.selected = Some(self.graph.add(kind, position));
        self.message = "Drag the card to arrange it, then connect its dots.".into();
        self.drag = None;
        cx.notify();
    }

    fn delete_selected(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self.selected.take() {
            self.graph.remove(id);
            self.drag = None;
            self.message = "Node removed with its connections.".into();
            cx.notify();
        }
    }

    fn fit(&mut self, cx: &mut Context<Self>) {
        self.drag = None;
        if self.graph.nodes.is_empty() {
            self.viewport = Viewport {
                pan: point(48., 80.),
                zoom: 1.,
            };
        } else {
            let left = self
                .graph
                .nodes
                .iter()
                .map(|n| n.position.x)
                .fold(f32::INFINITY, f32::min);
            let top = self
                .graph
                .nodes
                .iter()
                .map(|n| n.position.y)
                .fold(f32::INFINITY, f32::min);
            let right = self
                .graph
                .nodes
                .iter()
                .map(|n| n.position.x + WIDTH)
                .fold(f32::NEG_INFINITY, f32::max);
            let bottom = self
                .graph
                .nodes
                .iter()
                .map(|n| n.position.y + HEIGHT)
                .fold(f32::NEG_INFINITY, f32::max);
            let bounds = self.bounds.get();
            let w = f32::from(bounds.size.width);
            let h = f32::from(bounds.size.height);
            let zoom = ((w - 96.) / (right - left))
                .min((h - 96.) / (bottom - top))
                .clamp(MIN_ZOOM, 1.);
            self.viewport = Viewport {
                zoom,
                pan: point(
                    (w - (right + left) * zoom) / 2.,
                    (h - (bottom + top) * zoom) / 2.,
                ),
            };
        }
        cx.notify();
    }

    fn zoom(&mut self, factor: f32, cx: &mut Context<Self>) {
        let bounds = self.bounds.get();
        self.viewport.zoom_at(
            point(
                f32::from(bounds.size.width) / 2.,
                f32::from(bounds.size.height) / 2.,
            ),
            self.viewport.zoom * factor,
        );
        self.drag = None;
        cx.notify();
    }

    fn card(&self, node: &Node, palette: Palette) -> impl IntoElement {
        let zoom = self.viewport.zoom;
        let p = self.viewport.screen(node.position);
        let selected = self.selected == Some(node.id);
        div()
            .absolute()
            .left(px(p.x))
            .top(px(p.y))
            .w(px(WIDTH * zoom))
            .h(px(HEIGHT * zoom))
            .rounded(px(10. * zoom))
            .border_1()
            .border_color(if selected { palette.accent } else { palette.node_border })
            .bg(palette.node)
            .text_color(palette.text)
            .cursor(CursorStyle::OpenHand)
            .child(
                div()
                    .absolute()
                    .left(px(16. * zoom))
                    .top(px(12. * zoom))
                    .text_size(px(14. * zoom))
                    .child(format!("{} {}", node.kind.label(), node.id + 1)),
            )
            .child(
                div()
                    .absolute()
                    .left(px(16. * zoom))
                    .top(px(35. * zoom))
                    .text_size(px(11. * zoom))
                    .text_color(palette.muted)
                    .child(node.kind.caption()),
            )
            .children(
                node.kind
                    .inputs()
                    .iter()
                    .enumerate()
                    .map(|(index, (ty, label))| {
                        let y = (78. + index as f32 * 28.) * zoom;
                        div()
                            .absolute()
                            .left(px(-6. * zoom))
                            .top(px(y - 6. * zoom))
                            .w(px(WIDTH * zoom))
                            .h(px(12. * zoom))
                            .child(
                                div()
                                    .absolute()
                                    .size(px(12. * zoom))
                                    .rounded_full()
                                    .bg(rgb(ty.color()))
                                    .cursor_pointer(),
                            )
                            .child(
                                div()
                                    .absolute()
                                    .left(px(21. * zoom))
                                    .top(px(-2. * zoom))
                                    .text_size(px(11. * zoom))
                                    .text_color(palette.strong_text)
                                    .child(*label),
                            )
                    }),
            )
            .when_some(node.kind.output(), |card, ty| {
                card.child(
                    div()
                        .absolute()
                        .right(px(-6. * zoom))
                        .top(px(72. * zoom))
                        .size(px(12. * zoom))
                        .rounded_full()
                        .bg(rgb(ty.color()))
                        .cursor_pointer(),
                )
                .child(
                    div()
                        .absolute()
                        .right(px(15. * zoom))
                        .top(px(70. * zoom))
                        .text_size(px(11. * zoom))
                        .text_color(palette.strong_text)
                        .child(match ty {
                            DataType::Text => "Text",
                            DataType::Image => "Image",
                        }),
                )
            })
    }

    fn wires(&self, palette: Palette) -> impl IntoElement {
        let record = self.bounds.clone();
        let viewport = self.viewport;
        let mut wires: Vec<_> = self
            .graph
            .edges
            .iter()
            .filter_map(|edge| {
                let source = self.graph.node(edge.source)?;
                let target = self.graph.node(edge.target)?;
                Some((
                    viewport.screen(Self::port_position(source, None)),
                    viewport.screen(Self::port_position(target, Some(edge.input))),
                    source.kind.output()?.color(),
                ))
            })
            .collect();
        if let Some(Drag::Wire { port, pointer }) = self.drag {
            if let Some(node) = self.graph.node(port.node) {
                let start = viewport.screen(Self::port_position(node, port.input));
                let (from, to) = if port.input.is_some() {
                    (pointer, start)
                } else {
                    (start, pointer)
                };
                wires.push((from, to, 0x8aa6ff));
            }
        }
        canvas(
            move |bounds, _, _| record.set(bounds),
            move |bounds, (), window, _| {
                window.with_content_mask(Some(gpui::ContentMask { bounds }), |window| {
                    // Draw only the visible part of the grid, regardless of how far the user pans.
                    let gap = 32. * viewport.zoom;
                    let mut x = viewport.pan.x.rem_euclid(gap);
                    while x < f32::from(bounds.size.width) {
                        let mut y = viewport.pan.y.rem_euclid(gap);
                        while y < f32::from(bounds.size.height) {
                            window.paint_quad(gpui::fill(
                                Bounds::new(
                                    bounds.origin + point(px(x), px(y)),
                                    size(px(1.5), px(1.5)),
                                ),
                                palette.grid,
                            ));
                            y += gap;
                        }
                        x += gap;
                    }
                    for (from, to, color) in wires {
                        let from = bounds.origin + point(px(from.x), px(from.y));
                        let to = bounds.origin + point(px(to.x), px(to.y));
                        let bend = ((f32::from(to.x - from.x).abs() * 0.5)
                            .max(60. * viewport.zoom))
                        .min(250.);
                        let mut path = PathBuilder::stroke(px(2.));
                        path.move_to(from);
                        path.cubic_bezier_to(
                            to,
                            from + point(px(bend), px(0.)),
                            to - point(px(bend), px(0.)),
                        );
                        if let Ok(path) = path.build() {
                            window.paint_path(path, rgb(color));
                        }
                    }
                });
            },
        )
        .absolute()
        .size_full()
    }
}

impl Render for WorkflowCanvas {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().flex_1().min_w_0().h_full().flex().flex_col()
            .child(div().flex().items_center().gap_2().p_3().flex_wrap()
                .child(div().text_sm().child("Add node"))
                .children(Kind::ALL.into_iter().enumerate().map(|(index, kind)| {
                    Button::new(("add-workflow-node", index)).small().label(kind.label())
                        .on_click(cx.listener(move |view, _, _, cx| view.add(kind, cx)))
                }))
                .child(div().flex_1())
                .child(Button::new("workflow-disconnect").small().label("Disconnect")
                    .disabled(!self.selected.is_some_and(|id| self.graph.edges.iter().any(|e| e.source == id || e.target == id)))
                    .on_click(cx.listener(|view, _, _, cx| {
                        if let Some(id) = view.selected {
                            view.graph.edges.retain(|e| e.source != id && e.target != id);
                            view.message = "Node disconnected.".into();
                            cx.notify();
                        }
                    })))
                .child(Button::new("workflow-remove").small().label("Remove").disabled(self.selected.is_none())
                    .on_click(cx.listener(|view, _, _, cx| view.delete_selected(cx)))))
            .child(div().px_3().pb_2().text_xs().text_color(palette(cx).muted)
                .child("Workflow canvas · Arrange and connect nodes. Configuration and execution are coming later."))
            .child(div().id("workflow-canvas").track_focus(&self.focus)
                .relative().flex_1().min_h_0().overflow_hidden().bg(palette(cx).canvas).cursor(CursorStyle::OpenHand)
                .on_mouse_down(MouseButton::Left, cx.listener(Self::begin))
                .on_mouse_down(MouseButton::Middle, cx.listener(Self::begin))
                .on_mouse_move(cx.listener(|view, event, _, cx| view.motion(event, cx)))
                .on_mouse_up(MouseButton::Left, cx.listener(|view, event: &gpui::MouseUpEvent, _, cx| view.finish(event.position, cx)))
                .on_mouse_up(MouseButton::Middle, cx.listener(|view, event: &gpui::MouseUpEvent, _, cx| view.finish(event.position, cx)))
                .on_mouse_up_out(MouseButton::Left, cx.listener(|view, _, _, cx| { view.drag = None; cx.notify(); }))
                .on_mouse_up_out(MouseButton::Middle, cx.listener(|view, _, _, cx| { view.drag = None; cx.notify(); }))
                .on_scroll_wheel(cx.listener(|view, event: &gpui::ScrollWheelEvent, _, cx| {
                    let delta = event.delta.pixel_delta(px(24.));
                    if event.modifiers.control || event.modifiers.platform {
                        view.viewport.zoom_at(view.local(event.position), view.viewport.zoom * (-f32::from(delta.y) * 0.005).exp());
                    } else {
                        view.viewport.pan.x += f32::from(delta.x);
                        view.viewport.pan.y += f32::from(delta.y);
                    }
                    view.drag = None;
                    cx.stop_propagation();
                    cx.notify();
                }))
                .on_key_down(cx.listener(|view, event: &gpui::KeyDownEvent, _, cx| {
                    match event.keystroke.key.as_str() {
                        "backspace" | "delete" => view.delete_selected(cx),
                        "escape" => { view.drag = None; view.selected = None; view.message = "Drag between matching dots to connect nodes.".into(); cx.notify(); },
                        _ => return,
                    }
                    cx.stop_propagation();
                }))
                .child(self.wires(palette(cx)))
                .children(self.graph.nodes.iter().map(|node| self.card(node, palette(cx)))))
            .child(div().flex().flex_wrap().items_center().gap_2().p_3().text_xs().text_color(palette(cx).muted)
                .child(Button::new("workflow-zoom-out").small().label("−").tooltip("Zoom out")
                    .disabled(self.viewport.zoom <= MIN_ZOOM).on_click(cx.listener(|view, _, _, cx| view.zoom(1. / 1.2, cx))))
                .child(format!("{:.0}%", self.viewport.zoom * 100.))
                .child(Button::new("workflow-zoom-in").small().label("+").tooltip("Zoom in")
                    .disabled(self.viewport.zoom >= MAX_ZOOM).on_click(cx.listener(|view, _, _, cx| view.zoom(1.2, cx))))
                .child(Button::new("workflow-fit").small().label("Fit all").on_click(cx.listener(|view, _, _, cx| view.fit(cx))))
                .child("Drag background / scroll to pan · ⌘ or Ctrl + scroll to zoom"))
            .child(div().px_3().pb_3().text_xs().text_color(palette(cx).message).child(self.message.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(node: usize) -> Port {
        Port { node, input: None }
    }
    fn input(node: usize, index: usize) -> Port {
        Port {
            node,
            input: Some(index),
        }
    }

    #[test]
    fn typed_connections_reject_duplicates_cycles_and_invalid_ports() {
        let mut graph = Graph::default();
        let prompt = graph.add(Kind::Prompt, point(0., 0.));
        let a = graph.add(Kind::Generate, point(0., 0.));
        let b = graph.add(Kind::Generate, point(0., 0.));
        let result = graph.add(Kind::Output, point(0., 0.));
        assert!(graph.connect(output(prompt), input(a, 0)).is_ok());
        assert!(graph.connect(output(prompt), input(a, 0)).is_err());
        assert!(graph.connect(output(prompt), input(a, 1)).is_err());
        assert!(graph.connect(output(a), input(a, 1)).is_err());
        assert!(graph.connect(output(a), input(b, 1)).is_ok());
        assert!(graph.connect(output(b), input(a, 1)).is_err());
        assert!(graph.connect(input(result, 0), output(b)).is_ok());
        assert!(graph.connect(output(result), input(a, 1)).is_err());
        assert!(graph.connect(output(a), input(b, 5)).is_err());
        assert!(graph.connect(output(999), input(b, 1)).is_err());
        assert!(graph.connect(output(a), output(b)).is_err());
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn replacing_single_inputs_and_deleting_nodes_preserves_other_edges() {
        let mut graph = Graph::default();
        let a = graph.add(Kind::Prompt, point(0., 0.));
        let b = graph.add(Kind::Prompt, point(0., 0.));
        let step = graph.add(Kind::Generate, point(0., 0.));
        let result = graph.add(Kind::Output, point(0., 0.));
        graph.connect(output(a), input(step, 0)).unwrap();
        graph.connect(output(step), input(result, 0)).unwrap();
        graph.connect(output(b), input(step, 0)).unwrap();
        assert_eq!(graph.edges.len(), 2);
        assert!(!graph.edges.iter().any(|e| e.source == a));
        graph.remove(b);
        assert_eq!(
            graph.edges,
            vec![Edge {
                source: step,
                target: result,
                input: 0
            }]
        );
        graph.remove(step);
        assert!(graph.edges.is_empty());
        assert_ne!(graph.add(Kind::Image, point(0., 0.)), step);
    }

    #[test]
    fn image_inputs_keep_connection_order_and_enforce_reference_limit() {
        let mut graph = Graph::default();
        let step = graph.add(Kind::Generate, point(0., 0.));
        let images: Vec<_> = (0..11)
            .map(|_| graph.add(Kind::Image, point(0., 0.)))
            .collect();
        for &image in &images[..10] {
            graph.connect(output(image), input(step, 1)).unwrap();
        }
        assert!(graph.connect(output(images[10]), input(step, 1)).is_err());
        assert_eq!(
            graph.edges.iter().map(|e| e.source).collect::<Vec<_>>(),
            images[..10]
        );
    }

    #[test]
    fn zoom_preserves_pointer_anchor_and_clamps_scale() {
        let mut viewport = Viewport {
            pan: point(-12400., 8430.),
            zoom: 0.8,
        };
        let anchor = point(423., 312.);
        let world = viewport.world(anchor);
        for zoom in [1.6, 100., 0.001, 1.] {
            viewport.zoom_at(anchor, zoom);
            let actual = viewport.screen(world);
            assert!((actual.x - anchor.x).abs() < 0.01);
            assert!((actual.y - anchor.y).abs() < 0.01);
            assert!((MIN_ZOOM..=MAX_ZOOM).contains(&viewport.zoom));
        }
    }
}
