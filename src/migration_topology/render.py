"""Render a migration description as an SVG topology diagram.

Null ports are drawn as ``unlabeled``. Null move orders are drawn as
``not recorded``. Nothing in this module invents interface names, addresses,
or service instances.
"""

from __future__ import annotations

from html import escape

from migration_topology.model import Device, Link, Migration

FONT = "DejaVu Sans, Segoe UI, Helvetica, Arial, sans-serif"
WIDTH = 1280
MARGIN = 24
GUTTER = 116

PAPER = "#f3f0e8"
INK = "#1c1917"
MUTED = "#57534e"
FAINT = "#8a8178"
CARD = "#fffdf8"
RULE = "#e3dcd1"
SOURCE = "#1b3a4b"
TARGET = "#115e59"
AMBER = "#9a3412"
AMBER_BG = "#fff4e6"
AMBER_LINE = "#f0b27a"
HEADER = "#172033"
WHITE = "#ffffff"
CWM = "#312e81"
NSO_GREEN = "#14532d"
STAGE_COLORS = ("#b45309", "#1d4ed8", "#0f766e", "#6d28d9")


def render_svg(migration: Migration) -> str:
    """Return a complete SVG document for ``migration``."""
    drawing = _Drawing()
    layout = _layout(migration)
    _draw(drawing, migration, layout)
    height = layout["height"]
    title = escape(migration.title)
    desc = escape(_description(migration))
    body = "\n".join(drawing.chunks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{_num(height)}" '
        f'viewBox="0 0 {WIDTH} {_num(height)}" role="img" font-family="{FONT}">\n'
        f"<title>{title}</title>\n"
        f"<desc>{desc}</desc>\n"
        "<defs>\n"
        '  <marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{AMBER}"/></marker>\n'
        '  <marker id="arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>\n'
        "</defs>\n"
        f"{body}\n"
        "</svg>\n"
    )


def _description(migration: Migration) -> str:
    source = _device_phrase(migration.source)
    target = _device_phrase(migration.target)
    stages = ", ".join(f"{stage.order} {stage.name}" for stage in migration.stages)
    ports = "Interface names that are missing are unlabeled."
    return (
        f"{migration.title}. Current state: {source} with uplinks into "
        f"{migration.network.label}. Target state: those links on {target}. "
        f"Controller {migration.controller.name}. "
        f"{migration.controller.workflow_manager.name} runs the stages. "
        f"{migration.controller.nso.name} commits each stage template. "
        f"Stages: {stages}. {ports}"
    )


def _device_phrase(device: Device) -> str:
    head = f"{device.vendor} {device.name}" if device.vendor else device.name
    return f"{head} ({device.os})"


def _layout(migration: Migration) -> dict[str, float | int]:
    card_w = (WIDTH - 2 * MARGIN - GUTTER) / 2
    pill_w = _pill_width(card_w, len(migration.links))
    pill_inner = max(pill_w - 20, 40)
    pill_h = _pill_height(migration, pill_inner)
    note_lines = _wrap(migration.network.note or "", card_w - 48, 12)
    device_h = max(_device_height(migration.source), _device_height(migration.target))
    top = 16
    kicker = 22
    gap_after_kicker = 10
    note_h = len(note_lines) * 16
    gap_after_note = 12 if note_lines else 4
    ellipse_h = 64
    gap_after_ellipse = 36
    gap_after_pills = 36 if migration.links else 8
    link_block = pill_h if migration.links else 28
    bottom = 18
    card_h = (
        top
        + kicker
        + gap_after_kicker
        + note_h
        + gap_after_note
        + ellipse_h
        + gap_after_ellipse
        + link_block
        + gap_after_pills
        + device_h
        + bottom
    )

    header_h = 92
    strip_y = header_h + 14
    strip_h = 42
    card_y = strip_y + strip_h + 16

    link_notes = _link_note_lines(migration, WIDTH - 2 * MARGIN)
    notes_h = (len(link_notes) * 18 + 8) if link_notes else 0
    notes_gap = 14 if link_notes else 0

    controller_y = card_y + card_h + notes_gap + notes_h + 16
    controller_h = _controller_height(migration)
    height = controller_y + controller_h + MARGIN

    return {
        "card_w": card_w,
        "card_h": card_h,
        "card_y": card_y,
        "pill_w": pill_w,
        "pill_h": pill_h,
        "device_h": device_h,
        "ellipse_h": ellipse_h,
        "note_h": note_h,
        "strip_y": strip_y,
        "strip_h": strip_h,
        "notes_y": card_y + card_h + notes_gap,
        "notes_h": notes_h,
        "controller_y": controller_y,
        "controller_h": controller_h,
        "height": height,
        "top": top,
        "kicker": kicker,
        "gap_after_kicker": gap_after_kicker,
        "gap_after_ellipse": gap_after_ellipse,
        "gap_after_note": gap_after_note,
        "gap_after_pills": gap_after_pills,
    }


def _pill_width(card_w: float, count: int) -> float:
    if count <= 0:
        return 0
    gap = 16
    available = card_w - 32
    natural = 198
    total = count * natural + (count - 1) * gap
    if total <= available:
        return natural
    return max(96, (available - (count - 1) * gap) / count)


def _pill_height(migration: Migration, inner_w: float) -> float:
    if not migration.links:
        return 0
    tallest = 0.0
    for link in migration.links:
        for port in (link.source_port, link.target_port):
            name_n = len(_wrap(link.name, inner_w, 13)) or 1
            port_n = len(_wrap(_port_label(port), inner_w, 15)) or 1
            order_n = len(_wrap(_order_label(link), inner_w, 11)) or 1
            tallest = max(tallest, 12 + name_n * 17 + 14 + port_n * 18 + order_n * 15 + 10)
    return tallest


def _device_height(device: Device) -> float:
    height = 14 + 26 + 20 + 14
    if device.vendor:
        height += 16
    if device.nso_device_name_specified:
        height += 18
    return height


def _controller_height(migration: Migration) -> float:
    note_lines = _wrap(migration.controller.note or "", WIDTH - 2 * MARGIN - 40, 13)
    blocked = _blocked_lines(migration, WIDTH - 2 * MARGIN - 64)
    cursor = 18 + 20 + 30
    if note_lines:
        cursor += len(note_lines) * 18 + 8
    cursor += 8 + _stage_row_height(migration) + 12
    if blocked:
        cursor += 16 + len(blocked) * 18 + 8
    return cursor + 18


def _stage_row_height(migration: Migration) -> float:
    return max(
        _component_box_height(migration.controller.workflow_manager, 210, commits=False),
        _component_box_height(migration.controller.nso, 236, commits=True),
        _stage_stack_height(migration),
    )


def _component_box_height(component, width: float, *, commits: bool) -> float:
    height = 14.0
    if component.short_name and component.short_name != component.name:
        height += 24
        height += 16 * len(_wrap(component.name, width - 24, 12) or [component.name])
    else:
        height += 26
    height += 4
    height += 16 * len(_wrap(component.role, width - 24, 12) or [component.role])
    if commits:
        height += 6 + 16 + 32
    return height + 16


def _stage_stack_height(migration: Migration) -> float:
    inner = _stage_inner_width(len(migration.stages))
    tallest = 0.0
    for stage in migration.stages:
        name_n = max(1, len(_wrap(stage.name, inner, 13)))
        kind_n = max(1, len(_wrap(stage.template_kind, inner, 12)))
        tallest = max(tallest, 14 + 22 + name_n * 17 + 6 + kind_n * 16 + 14)
    return tallest


def _stage_inner_width(count: int) -> float:
    outer = _stages_width()
    gap = 14
    count = max(count, 1)
    card = (outer - (count - 1) * gap) / count
    return max(40, card - 16)


def _stages_width() -> float:
    inner = WIDTH - 2 * MARGIN - 32
    return inner - 210 - 236 - 28 - 28


def _draw(drawing: _Drawing, migration: Migration, layout: dict) -> None:
    height = layout["height"]
    drawing.rect(0, 0, WIDTH, height, PAPER)
    _draw_header(drawing, migration)
    _draw_stage_strip(drawing, migration, layout["strip_y"], layout["strip_h"])
    card_w = layout["card_w"]
    left_x = MARGIN
    right_x = MARGIN + card_w + GUTTER
    _draw_state(
        drawing,
        migration,
        layout,
        x=left_x,
        side="source",
        kicker="CURRENT",
        caption="uplinks on the source",
        device=migration.source,
        fill=SOURCE,
    )
    _draw_state(
        drawing,
        migration,
        layout,
        x=right_x,
        side="target",
        kicker="TARGET",
        caption="uplinks on the target",
        device=migration.target,
        fill=TARGET,
    )
    _draw_move_arrow(drawing, migration, layout, left_x + card_w, right_x)
    _draw_link_notes(drawing, migration, layout)
    _draw_controller(drawing, migration, layout)


def _draw_header(drawing: _Drawing, migration: Migration) -> None:
    drawing.rect(0, 0, WIDTH, 92, HEADER)
    drawing.text(MARGIN, 20, migration.title, 22, WHITE, 700)
    subtitle = _device_phrase(migration.source) + "   →   " + _device_phrase(migration.target)
    if migration.kind:
        subtitle = f"{migration.kind}   ·   {subtitle}"
    drawing.text(MARGIN, 54, subtitle, 14, "#d6deea", 400)
    if migration.status:
        _draw_status(drawing, migration.status)


def _draw_status(drawing: _Drawing, status: str) -> None:
    label = status.upper()
    chip_w = max(96, _text_width(label, 12) + 28)
    chip_h = 28
    chip_x = WIDTH - MARGIN - chip_w
    chip_y = 32
    blocked = status.lower() == "blocked"
    drawing.rect(chip_x, chip_y, chip_w, chip_h, "#7c2d12" if blocked else "#334155", rx=14)
    drawing.text(
        chip_x + chip_w / 2,
        chip_y + chip_h / 2,
        label,
        12,
        WHITE,
        700,
        anchor="middle",
        baseline="central",
    )


def _draw_stage_strip(drawing: _Drawing, migration: Migration, y: float, h: float) -> None:
    drawing.rect(MARGIN, y, WIDTH - 2 * MARGIN, h, CARD, stroke=RULE, rx=10)
    chain = "    →    ".join(f"{stage.order}   {stage.name}" for stage in migration.stages)
    drawing.text(WIDTH / 2, y + h / 2, chain, 15, INK, 600, anchor="middle", baseline="central")


def _draw_state(
    drawing: _Drawing,
    migration: Migration,
    layout: dict,
    *,
    x: float,
    side: str,
    kicker: str,
    caption: str,
    device: Device,
    fill: str,
) -> None:
    y = layout["card_y"]
    w = layout["card_w"]
    h = layout["card_h"]
    drawing.rect(x, y, w, h, CARD, stroke=RULE, sw=1.5, rx=16)
    cursor = y + layout["top"]
    drawing.text(x + 18, cursor, kicker, 12, MUTED, 700)
    drawing.text(x + w - 18, cursor, caption, 12, FAINT, 400, anchor="end")
    cursor += layout["kicker"] + layout["gap_after_kicker"]

    cx = x + w / 2
    note_lines = _wrap(migration.network.note or "", w - 48, 12)
    if note_lines:
        drawing.lines(cx, cursor, note_lines, 12, MUTED, anchor="middle", leading=16)
        cursor += layout["note_h"]
    cursor += layout["gap_after_note"]

    ellipse_h = layout["ellipse_h"]
    cy = cursor + ellipse_h / 2
    rx = min(w / 2 - 36, max(86, _text_width(migration.network.label, 16) / 2 + 28))
    ry = 28
    drawing.ellipse(cx, cy, rx, ry, "#ece7dc", "#78716c", 1.5)
    drawing.text(cx, cy, migration.network.label, 16, INK, 700, anchor="middle", baseline="central")
    cursor += ellipse_h + layout["gap_after_ellipse"]

    pills = _pill_boxes(x, w, len(migration.links), layout["pill_w"], layout["pill_h"], cursor)
    device_w = _device_width(w)
    if migration.links:
        device_y = cursor + layout["pill_h"] + layout["gap_after_pills"]
        _draw_cables(drawing, cx, cy + ry, rx, pills, device_y, device_w)
        for link, box in zip(migration.links, pills, strict=True):
            port = link.source_port if side == "source" else link.target_port
            _draw_pill(drawing, link, port, box, side)
        cursor = device_y
    else:
        drawing.line(cx, cy + ry, cx, cursor + 36, "#8a8175", 1.5, marker=None)
        drawing.text(cx, cursor, "no links recorded", 12, FAINT, 400, anchor="middle")
        cursor += 36
    device_x = x + (w - device_w) / 2
    _draw_device(drawing, device, device_x, cursor, device_w, layout["device_h"], fill)


def _device_width(card_w: float) -> float:
    return min(300, card_w - 56)


def _pill_boxes(
    card_x: float,
    card_w: float,
    count: int,
    pill_w: float,
    pill_h: float,
    y: float,
) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []
    gap = 16
    total = count * pill_w + (count - 1) * gap
    start = card_x + (card_w - total) / 2
    return [(start + index * (pill_w + gap), y, pill_w, pill_h) for index in range(count)]


def _draw_cables(
    drawing: _Drawing,
    cx: float,
    ellipse_bottom: float,
    rx: float,
    pills: list[tuple[float, float, float, float]],
    device_y: float,
    device_w: float,
) -> None:
    count = len(pills)
    if count == 0:
        return
    for index, (px, py, pw, ph) in enumerate(pills):
        spread = 0 if count == 1 else (index / (count - 1)) * 2 - 1
        start_x = cx + spread * rx * 0.62
        pill_cx = px + pw / 2
        drawing.line(start_x, ellipse_bottom, pill_cx, py, "#8a8175", 1.75)
        drawing.circle(start_x, ellipse_bottom, 3.2, "#8a8175")
        end_spread = 0 if count == 1 else (index / (count - 1)) * 2 - 1
        end_x = cx + end_spread * (device_w * 0.32)
        drawing.line(pill_cx, py + ph, end_x, device_y, "#8a8175", 1.75)
        drawing.circle(end_x, device_y, 3.2, "#8a8175")
        drawing.circle(pill_cx, py, 3.2, "#8a8175")
        drawing.circle(pill_cx, py + ph, 3.2, "#8a8175")


def _draw_pill(
    drawing: _Drawing,
    link: Link,
    port: str | None,
    box: tuple[float, float, float, float],
    side: str,
) -> None:
    x, y, w, h = box
    unknown = port is None
    drawing.open_group(link.note)
    drawing.rect(
        x,
        y,
        w,
        h,
        AMBER_BG if unknown else WHITE,
        stroke=AMBER_LINE if unknown else RULE,
        sw=1.5,
        rx=10,
    )
    inner = w - 20
    cx = x + w / 2
    cursor = y + 10
    name_lines = _wrap(link.name, inner, 13) or [link.name]
    cursor += drawing.lines(cx, cursor, name_lines, 13, INK, 650, anchor="middle", leading=17)
    side_label = "source port" if side == "source" else "target port"
    drawing.text(cx, cursor, side_label, 10, FAINT, 400, anchor="middle")
    cursor += 14
    port_lines = _wrap(_port_label(port), inner, 15) or [_port_label(port)]
    cursor += drawing.lines(
        cx,
        cursor,
        port_lines,
        15,
        AMBER if unknown else INK,
        700,
        anchor="middle",
        leading=18,
    )
    order_lines = _wrap(_order_label(link), inner, 11) or [_order_label(link)]
    drawing.lines(
        cx,
        cursor + 2,
        order_lines,
        11,
        AMBER if link.move_order is None else MUTED,
        400,
        anchor="middle",
        leading=15,
    )
    drawing.close_group()


def _draw_device(
    drawing: _Drawing,
    device: Device,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
) -> None:
    drawing.rect(x, y, w, h, fill, rx=12)
    cursor = y + 12
    cx = x + w / 2
    if device.vendor:
        drawing.text(cx, cursor, device.vendor.upper(), 11, "#d6e2ea", 700, anchor="middle")
        cursor += 16
    drawing.text(cx, cursor, device.name, 20, WHITE, 700, anchor="middle")
    cursor += 26
    drawing.text(cx, cursor, device.os, 14, "#d6e2ea", 400, anchor="middle")
    cursor += 20
    if device.nso_device_name_specified:
        shown = device.nso_device_name or "not recorded"
        label = f"NSO device name {shown}" if device.nso_device_name is None else f"NSO device {shown}"
        drawing.text(cx, cursor, label, 11, "#f3d3b0" if device.nso_device_name is None else "#d6e2ea", 400, anchor="middle")


def _draw_move_arrow(
    drawing: _Drawing,
    migration: Migration,
    layout: dict,
    left: float,
    right: float,
) -> None:
    first = migration.stages[0]
    cx = (left + right) / 2
    pill_y = (
        layout["card_y"]
        + layout["top"]
        + layout["kicker"]
        + layout["gap_after_kicker"]
        + layout["note_h"]
        + layout["gap_after_note"]
        + layout["ellipse_h"]
        + layout["gap_after_ellipse"]
    )
    cy = pill_y + max(layout["pill_h"], 28) / 2
    drawing.line(cx - 36, cy, cx + 28, cy, AMBER, 2, marker="url(#arrow-amber)")
    name_lines = _wrap(first.name, GUTTER - 8, 11) or [first.name]
    drawing.lines(cx, cy - 18 - 14 * (len(name_lines) - 1), name_lines, 11, AMBER, 700, anchor="middle", leading=14)
    drawing.text(cx, cy + 10, "first stage", 11, AMBER, 400, anchor="middle")


def _draw_link_notes(drawing: _Drawing, migration: Migration, layout: dict) -> None:
    lines = _link_note_lines(migration, WIDTH - 2 * MARGIN)
    if not lines:
        return
    y = layout["notes_y"]
    drawing.lines(MARGIN, y, lines, 12, MUTED, leading=18)


def _link_note_lines(migration: Migration, width: float) -> list[str]:
    lines: list[str] = []
    for link in migration.links:
        if not link.note:
            continue
        lines.extend(_wrap(f"{link.name}: {link.note}", width, 12))
    return lines


def _draw_controller(drawing: _Drawing, migration: Migration, layout: dict) -> None:
    x = MARGIN
    y = layout["controller_y"]
    w = WIDTH - 2 * MARGIN
    h = layout["controller_h"]
    drawing.rect(x, y, w, h, CARD, stroke=RULE, sw=1.5, rx=16)
    cursor = y + 18
    drawing.text(x + 18, cursor, "CONTROLLER", 12, MUTED, 700)
    cursor += 20
    heading = migration.controller.name
    if migration.controller.short_name:
        heading = f"{heading} ({migration.controller.short_name})"
    drawing.text(x + 18, cursor, heading, 20, INK, 700)
    cursor += 30
    note_lines = _wrap(migration.controller.note or "", w - 40, 13)
    if note_lines:
        cursor += drawing.lines(x + 18, cursor, note_lines, 13, MUTED, leading=18)
        cursor += 8
    cursor += 8
    row_h = _stage_row_height(migration)
    _draw_control_row(drawing, migration, x + 16, cursor, w - 32, row_h)
    cursor += row_h + 12
    _draw_blocked(drawing, migration, x + 16, cursor, w - 32)


def _draw_control_row(
    drawing: _Drawing,
    migration: Migration,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    cwm_w = 210
    nso_w = 236
    arrow_gap = 28
    stages_w = w - cwm_w - nso_w - 2 * arrow_gap
    cwm_x = x
    stages_x = x + cwm_w + arrow_gap
    nso_x = stages_x + stages_w + arrow_gap
    _draw_component_box(
        drawing,
        migration.controller.workflow_manager,
        cwm_x,
        y,
        cwm_w,
        h,
        CWM,
        commits_to=None,
    )
    stage_h = _stage_stack_height(migration)
    stage_y = y + max(0, (h - stage_h) / 2)
    _draw_stages(drawing, migration, stages_x, stage_y, stages_w, stage_h)
    _draw_component_box(
        drawing,
        migration.controller.nso,
        nso_x,
        y,
        nso_w,
        h,
        NSO_GREEN,
        commits_to=(migration.source.name, migration.target.name),
    )
    mid = stage_y + stage_h / 2
    drawing.line(cwm_x + cwm_w, mid, stages_x, mid, MUTED, 1.5, marker="url(#arrow-ink)")
    drawing.line(stages_x + stages_w, mid, nso_x, mid, MUTED, 1.5, marker="url(#arrow-ink)")


def _draw_component_box(
    drawing: _Drawing,
    component,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    commits_to: tuple[str, str] | None,
) -> None:
    drawing.rect(x, y, w, h, fill, rx=12)
    cursor = y + 14
    cx = x + w / 2
    if component.short_name and component.short_name != component.name:
        drawing.text(cx, cursor, component.short_name, 18, WHITE, 700, anchor="middle")
        cursor += 24
        for line in _wrap(component.name, w - 24, 12):
            drawing.text(cx, cursor, line, 12, "#e0e7ff", 400, anchor="middle")
            cursor += 16
    else:
        drawing.text(cx, cursor, component.name, 18, WHITE, 700, anchor="middle")
        cursor += 26
    cursor += 4
    for line in _wrap(component.role, w - 24, 12):
        drawing.text(cx, cursor, line, 12, "#e7e5e4", 400, anchor="middle")
        cursor += 16
    if commits_to:
        cursor += 6
        drawing.text(cx, cursor, "commits to", 11, "#d1fae5", 700, anchor="middle")
        cursor += 16
        drawing.text(cx, cursor, commits_to[0], 13, WHITE, 650, anchor="middle")
        cursor += 16
        drawing.text(cx, cursor, commits_to[1], 13, WHITE, 650, anchor="middle")


def _draw_stages(drawing: _Drawing, migration: Migration, x: float, y: float, w: float, h: float) -> None:
    count = len(migration.stages)
    gap = 14
    card_w = (w - (count - 1) * gap) / count
    inner = card_w - 16
    for index, stage in enumerate(migration.stages):
        sx = x + index * (card_w + gap)
        color = STAGE_COLORS[index % len(STAGE_COLORS)]
        drawing.open_group(stage.summary)
        drawing.rect(sx, y, card_w, h, "#fffcf7", stroke=RULE, rx=10)
        cx = sx + card_w / 2
        badge_y = y + 14
        drawing.circle(cx, badge_y + 9, 10, color)
        drawing.text(cx, badge_y + 9, str(stage.order), 11, WHITE, 700, anchor="middle", baseline="central")
        cursor = badge_y + 24
        name_lines = _wrap(stage.name, inner, 13) or [stage.name]
        cursor += drawing.lines(cx, cursor, name_lines, 13, INK, 700, anchor="middle", leading=17)
        cursor += 4
        kind_lines = _wrap(stage.template_kind, inner, 12) or [stage.template_kind]
        drawing.lines(cx, cursor, kind_lines, 12, MUTED, 400, anchor="middle", leading=16)
        drawing.close_group()
        if index < count - 1:
            mid = y + h / 2
            drawing.line(sx + card_w + 1, mid, sx + card_w + gap - 1, mid, FAINT, 1.25, marker="url(#arrow-ink)")


def _draw_blocked(drawing: _Drawing, migration: Migration, x: float, y: float, w: float) -> None:
    lines = _blocked_lines(migration, w - 28)
    if not lines:
        return
    h = 16 + len(lines) * 18 + 8
    drawing.rect(x, y, w, h, AMBER_BG, stroke=AMBER_LINE, rx=10)
    drawing.lines(x + 14, y + 10, lines, 13, AMBER, 650, leading=18)


def _blocked_lines(migration: Migration, width: float) -> list[str]:
    sentences: list[str] = []
    if migration.blocked_on:
        sentences.append("Blocked on " + _and_join(list(migration.blocked_on)) + ".")
    if any(link.source_port is None or link.target_port is None for link in migration.links):
        sentences.append("Missing interface names are labeled unlabeled.")
    if any(link.move_order is None for link in migration.links):
        sentences.append("Missing move orders are labeled not recorded.")
    lines: list[str] = []
    for sentence in sentences:
        lines.extend(_wrap(sentence, width, 13))
    return lines


def _and_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _port_label(port: str | None) -> str:
    return "unlabeled" if port is None else port


def _order_label(link: Link) -> str:
    if link.move_order is None:
        return "move order not recorded"
    return f"move order {link.move_order}"


def _wrap(text: str, width: float, font_size: float) -> list[str]:
    if not text:
        return []
    max_chars = max(1, int(width / (font_size * 0.56)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        piece = word
        while _text_units(piece) > max_chars:
            lines.append(piece[:max_chars])
            piece = piece[max_chars:]
        trial = piece if not current else f"{current} {piece}"
        if _text_units(trial) <= max_chars:
            current = trial
        else:
            if current:
                lines.append(current)
            current = piece
    if current:
        lines.append(current)
    return lines


def _text_units(text: str) -> int:
    return len(text)


def _text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.56


class _Drawing:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "none",
        sw: float = 1,
        rx: float = 0,
    ) -> None:
        attrs = {
            "x": _num(x),
            "y": _num(y),
            "width": _num(w),
            "height": _num(h),
            "fill": fill,
        }
        if rx:
            attrs["rx"] = _num(rx)
            attrs["ry"] = _num(rx)
        if stroke != "none":
            attrs["stroke"] = stroke
            attrs["stroke-width"] = _num(sw)
        self._el("rect", attrs)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, fill: str, stroke: str, sw: float) -> None:
        self._el(
            "ellipse",
            {
                "cx": _num(cx),
                "cy": _num(cy),
                "rx": _num(rx),
                "ry": _num(ry),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": _num(sw),
            },
        )

    def circle(self, cx: float, cy: float, r: float, fill: str) -> None:
        self._el("circle", {"cx": _num(cx), "cy": _num(cy), "r": _num(r), "fill": fill})

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        sw: float,
        marker: str | None = None,
    ) -> None:
        attrs = {
            "x1": _num(x1),
            "y1": _num(y1),
            "x2": _num(x2),
            "y2": _num(y2),
            "stroke": stroke,
            "stroke-width": _num(sw),
            "stroke-linecap": "round",
        }
        if marker:
            attrs["marker-end"] = marker
        self._el("line", attrs)

    def text(
        self,
        x: float,
        y: float,
        content: str,
        size: float,
        fill: str,
        weight: int = 400,
        anchor: str = "start",
        baseline: str = "hanging",
    ) -> None:
        self._el(
            "text",
            {
                "x": _num(x),
                "y": _num(y),
                "fill": fill,
                "font-size": _num(size),
                "font-weight": str(weight),
                "text-anchor": anchor,
                "dominant-baseline": baseline,
            },
            escape(content),
        )

    def lines(
        self,
        x: float,
        y: float,
        texts: list[str],
        size: float,
        fill: str,
        weight: int = 400,
        anchor: str = "start",
        leading: float = 16,
    ) -> float:
        for index, content in enumerate(texts):
            self.text(x, y + index * leading, content, size, fill, weight, anchor)
        return len(texts) * leading

    def open_group(self, title: str | None = None) -> None:
        self.chunks.append("<g>")
        if title:
            self._el("title", {}, escape(title))

    def close_group(self) -> None:
        self.chunks.append("</g>")

    def _el(self, tag: str, attrs: dict[str, str], body: str | None = None) -> None:
        rendered = "".join(f' {key}="{_attr(value)}"' for key, value in attrs.items())
        if body is None:
            self.chunks.append(f"<{tag}{rendered}/>")
        else:
            self.chunks.append(f"<{tag}{rendered}>{body}</{tag}>")


def _attr(value: str) -> str:
    return escape(str(value), {'"': "&quot;"})


def _num(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.2f}"
