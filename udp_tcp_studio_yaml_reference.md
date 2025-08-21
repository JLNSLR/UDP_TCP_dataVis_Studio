# UDP/TCP Studio — YAML Reference

A compact reference for configuring the real-time plotting app.

---

## `app`

```yaml
app:
  title: "UDP/TCP Studio"
  fps: 60                        # UI update timer in Hz
  hide_timestamp_legend: true    # hide timestamp channel in legends
  drop_timestamp_channel: true   # never plot the timestamp channel
  crosshair: true                # default per-plot crosshair
  inspector: false               # default per-plot Inspector visibility
  crosshair_max_channels: 8      # rows shown in Inspector readout

  # Theme (window + plot colors)
  theme:
    name: fusion-light           # fusion-light | fusion-dark | native
    pg_background: "#0f0f12"     # plot canvas
    pg_foreground: "#e6e6e6"     # axes/labels
    stylesheet: |                # extra QSS (optional)
      QDockWidget::title { font-weight: 600; }
```

---

## `data_sources`

Define where vectors come from. Each emits a fixed-length float vector.

### Common (any type)

```yaml
- name: udp0
  type: udp | tcp_client | tcp_server
  length: 22              # expected vector size (optional for UDP, required for float32 TCP)
  pad_missing: true       # pad short frames with zeros
  drop_excess: true       # truncate longer frames

  # timestamping (choose one)
  timestamp_index: 21     # use element as timestamp…
  timestamp_units: s      # s | ms | us | ns  (default s)
  timestamp_scale: 1.0    # or explicit scale instead of units
  timestamp_zero: true    # subtract first timestamp (start at ~0 s)

  # OR if no timestamp channel exists:
  sample_rate_hz: 200     # synthesizes time: t = n / fs
```

### UDP

```yaml
- name: udp0
  type: udp
  bind_ip: "0.0.0.0"
  port: 5005
  format: float32_le      # binary little-endian float32
  # If sender sends ASCII instead, the parser will try text fallback automatically.
```

### TCP client

```yaml
- name: tcp0
  type: tcp_client
  host: "127.0.0.1"
  port: 7000
  format: float32_le_fixed   # requires 'length'
  length: 22
  # or:
  # format: ascii_line
  # line_delimiter: "\\n"
```

### TCP server

Accepts a single client and reads frames.

```yaml
- name: tcps
  type: tcp_server
  bind_ip: "0.0.0.0"
  port: 7001
  format: float32_le_fixed
  length: 22
  # or ascii_line + line_delimiter
```

---

## `channels`

Map human-readable names to vector indices.

### Shorthand range

```yaml
channels:
  - range: "ch0..ch21"     # inclusive, same prefix
```

### Explicit entries

```yaml
channels:
  - { name: ax, index: 0, unit: "g", scale: 0.5, offset: 0.0, color: "#4caf50", width: 2.0 }
  - { name: ts, index: 21, unit: "s" }
```

> `scale`/`offset` are applied to live data; `color` can be any Qt/pyqtgraph color (name or hex).

---

## `derived` (virtual channels)

Create channels from expressions (evaluated with `np` and `math` available).

```yaml
derived:
  - { name: mag, expr: "np.sqrt(ch0**2 + ch1**2 + ch2**2)" }
```

> Expressions see the **latest values** of real/derived channels as variables by name.

---

## `plots`

Each plot shows one source’s channels.

```yaml
plots:
  - title: "TCP Demo"
    source: tcp0                 # which data source drives it
    channels: ["ch0..ch5", "ts"] # names from `channels` (timestamp will be dropped if app.drop_timestamp_channel=true)
    color_cycle: ["#ff5722", "#03a9f4"]  # optional per-plot palette
    window_samples: 5000         # history buffer per curve before decimation
    decimate: 1                  # 1 = no decimation (set >1 at very high rates)
    grid: [true, true]           # show x/y grid
    y_range: [-1.0, 1.0]         # optional fixed Y
    follow_x: true               # auto-roll to keep newest data at right edge
    crosshair: true              # show crosshair lines
    inspector: false             # start with Inspector collapsed
    drop_timestamp_channel: true # per-plot override (otherwise app-level)
    hide_timestamp_legend: true  # per-plot override

    # link multiple plots’ X-ranges
    x_link_group: g1             # plots with same group share X range
    x_link_master: true          # this one drives the group

    # x-axis time base
    x_axis:
      mode: timestamp            # samples | time | timestamp
      window_s: 6.0              # span shown when following (sec). If omitted, inferred from median Δt.
      lock_right: true           # stick to latest on the right
      sample_rate_hz: 200        # only used for mode: time (fixed-rate)
```

**X-axis modes**

- `samples`: x = 0..N, rolls by sample index.  
- `time`: uses `sample_rate_hz` for x (seconds), no per-packet timestamp.  
- `timestamp`: uses a channel (`timestamp_index`) or source `sample_rate_hz` (synthetic) if no timestamp channel.

---

## `layout`

Dock arrangement (rows of titles).

```yaml
layout:
  rows:
    - ["TCP Demo", "UDP Demo"]
    - ["FFT", "Inspector"]
```

> You can also Save/Load dock layout at runtime (stored as `dock_layout.state` next to the config).

---

## `control_groups` & `controls` (TX UI)

Create UI controls whose values can be referenced in sender vectors.

```yaml
control_groups:
  - { name: drive, title: "Drive Params", apply_label: "Apply Drive" }
  - { name: flags, title: "Flags" }

controls:
  # Float control with slider+spinbox
  - id: amp
    group: drive
    type: float
    label: "Amplitude"
    min: 0.0
    max: 10.0
    step: 0.1
    slider_steps: 1000
    default: 1.0
    apply_required: true   # staged until group Apply is pressed

  # Bool (checkbox)
  - id: enable
    group: flags
    type: bool
    label: "Enable output"
    default: true
    apply_required: false  # apply instantly

  # Momentary button (pulses active value)
  - id: fire
    group: flags
    type: button
    label: "Fire"
    pulse_ms: 50           # reverts to 0 after this
```

---

## `senders` (UDP TX)

Build and send vectors at a fixed rate; elements can reference controls with `$id`.

```yaml
senders:
  - name: out0
    type: udp
    remote_ip: "127.0.0.1"
    remote_port: 6000
    rate_hz: 50
    format: float32_le
    vector:
      template:
        - $amp             # pulls from control ‘amp’ (active value)
        - $enable
        - 3.14
        - $fire
```

---

## Minimal working example

```yaml
app:
  title: "My Lab Scope"
  fps: 60
  hide_timestamp_legend: true
  drop_timestamp_channel: true
  crosshair: true
  inspector: false
  theme:
    name: fusion-light
    pg_background: "#0f0f12"
    pg_foreground: "#e6e6e6"

data_sources:
  - name: tcp0
    type: tcp_client
    host: "127.0.0.1"
    port: 7000
    format: float32_le_fixed
    length: 22
    timestamp_index: 21
    timestamp_units: s
    timestamp_zero: true

channels:
  - range: "ch0..ch21"

plots:
  - title: "TCP Demo"
    source: tcp0
    channels: ["ch0..ch5", "ch21"]
    window_samples: 5000
    grid: [true, true]
    x_axis: { mode: timestamp, window_s: 6.0, lock_right: true }
    follow_x: true
    inspector: false
    x_link_group: g1
    x_link_master: true

  - title: "Zoomed"
    source: tcp0
    channels: ["ch0", "ch1"]
    window_samples: 5000
    grid: [true, false]
    x_axis: { mode: timestamp, window_s: 2.0 }
    follow_x: false
    inspector: true
    x_link_group: g1

layout:
  rows:
    - ["TCP Demo"]
    - ["Zoomed"]

control_groups:
  - { name: drive, title: "Drive", apply_label: "Apply" }

controls:
  - { id: amp, group: drive, type: float, label: "Amplitude", min: 0, max: 5, step: 0.1, default: 1.0, apply_required: true }
  - { id: enable, group: drive, type: bool, label: "Enable", default: true, apply_required: false }
  - { id: fire, group: drive, type: button, label: "Fire", pulse_ms: 50 }

senders:
  - name: out0
    type: udp
    remote_ip: "127.0.0.1"
    remote_port: 6000
    rate_hz: 20
    format: float32_le
    vector:
      template: [ $amp, $enable, $fire, 0, 0, 0 ]
```

---

## Tips & gotchas

- **Range expansion**: `"ch0..ch21"` is inclusive; prefixes must match (e.g., `a0..a5` OK; `a0..b5` ❌).
- **Timestamp channel**:
  - Set `timestamp_index` to the element that carries time.
  - With `drop_timestamp_channel: true`, it won’t be drawn (avoids the diagonal line).
  - If you don’t have a timestamp channel, set `sample_rate_hz` (per-source) and use `x_axis.mode: time`.
- **Follow window**:
  - Set `plots[].x_axis.window_s` for a crisp, non-drifting follow; if omitted, the app infers it (median Δt × visible samples).
- **Hot reload**: editing `config.yaml` is picked up automatically; errors will show in the status bar.
- **Linked X**: pick **one** master per `x_link_group`; others mirror its X-range.
- **Senders**: `$var` pulls from **active** control values (use `apply_required: true` in a group with an `apply_label` if you want staged changes).
