"""Local review interface for observed Formula Drift tandem tracks."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs" / "runs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ACCENT = "#d7ff60"

st.set_page_config(page_title="DriftLens | Tandem review", page_icon="🏁", layout="wide")
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: #0d1013; color: #f1f4f5; font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: #0d1013; }
    [data-testid="stSidebar"] { background: #13181d; border-right: 1px solid #283036; }
    .block-container { max-width: 1540px; padding-top: 1.65rem; padding-bottom: 3rem; }
    h1,h2,h3 { font-family: 'Barlow Condensed','Arial Narrow',sans-serif!important; letter-spacing: .01em; }
    h1 { font-size: 3.6rem!important; line-height: .98!important; }
    h2 { font-size: 2rem!important; }
    .brand { font-family:'Barlow Condensed','Arial Narrow',sans-serif; font-weight:800; font-size:1.9rem; letter-spacing:.09em; }
    .brand span { color:#d7ff60; }
    .eyebrow { color:#d7ff60; font-size:.72rem; letter-spacing:.18em; font-weight:700; margin-bottom:1rem; }
    .hero { padding:1.9rem 2rem; border:1px solid #2f3b33; border-radius:14px;
      background:linear-gradient(115deg,#18211a 0%,#141a1f 57%,#11161a 100%); margin-bottom:1.4rem; }
    .hero h1 { margin:0 0 .75rem 0; max-width:800px; }
    .hero p { color:#aeb9c1; max-width:720px; line-height:1.65; margin-bottom:.15rem; }
    .pill { display:inline-block; border:1px solid #3a4434; padding:.32rem .65rem; border-radius:4px;
      font-size:.7rem; color:#d7ff60; margin:.95rem .35rem 0 0; letter-spacing:.05em; }
    [data-testid="stMetric"] { background:#161d23; border:1px solid #2c363e; padding:1rem 1.2rem; border-radius:9px; }
    [data-testid="stMetricLabel"] { color:#9cabb5; font-size:.75rem; }
    [data-testid="stMetricValue"] { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2rem; }
    .stTabs [data-baseweb="tab-list"] { gap:2rem; background:transparent; border-bottom:1px solid #2a343b; }
    .stTabs [data-baseweb="tab"] { font-size:.85rem; padding:1rem .15rem; color:#b3bfc7; }
    .stTabs [aria-selected="true"] { color:#d7ff60; }
    .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] { background:#d7ff60; color:#172014; border:none; font-weight:700; }
    .stButton>button,.stDownloadButton>button { border:1px solid #3b4852; border-radius:6px; }
    .stButton>button:hover,.stDownloadButton>button:hover { border-color:#d7ff60; color:#d7ff60; }
    [data-testid="stCaptionContainer"] { color:#9aabb7; }
    [data-testid="stExpander"] { border-color:#2d3941; }
    .small-label { color:#9aabb7; text-transform:uppercase; font-size:.65rem; letter-spacing:.14em; }
    .side-note { color:#9aa9b4; font-size:.8rem; line-height:1.6; }
    .footer { color:#647783; font-size:.75rem; padding-top:1.5rem; border-top:1px solid #253039; margin-top:2rem; }
    </style>""",
    unsafe_allow_html=True,
)


def project_path(value: str | Path, base: Path = ROOT) -> Path | None:
    """Only allow local paths resolving inside this independent project."""
    try:
        path = Path(value)
        result = (path if path.is_absolute() else base / path).resolve()
        result.relative_to(ROOT)
        return result
    except (ValueError, OSError, TypeError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def number(value: Any, default: float = 0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["1", "1.0", "true", "yes"])


def run_file(run_dir: Path, summary: dict, field: str, fallback: str) -> Path | None:
    files = summary.get("files") or {}
    path = project_path(files.get(field, fallback), run_dir)
    return path if path and path.is_file() else None


def discover_runs() -> list[tuple[Path, dict]]:
    results = []
    if not RUNS.exists():
        return results
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir() or folder.name.lower() in {"comparisons", "comparison"} or folder.name.startswith("."):
            continue
        safe = project_path(folder)
        if safe is None:
            continue
        summary = read_json(safe / "summary.json")
        if summary and summary.get("status", "complete") == "complete":
            results.append((safe, summary))
    return results


def discover_sources(catalog: dict) -> list[Path]:
    candidates = list(ROOT.glob("*"))
    for raw_dir in (ROOT / "data" / "raw", ROOT / "data" / "videos"):
        if raw_dir.exists():
            candidates.extend(raw_dir.rglob("*"))
    catalog_source = project_path((catalog.get("source") or {}).get("path", "longbeach2024_action.mp4"))
    if catalog_source:
        candidates.append(catalog_source)
    paths = set()
    for candidate in candidates:
        safe = project_path(candidate)
        if safe and safe.is_file() and safe.suffix.lower() in VIDEO_SUFFIXES:
            paths.add(safe)
    default = ROOT / "longbeach2024_action.mp4"
    return sorted(paths, key=lambda path: (path != default, str(path).lower()))


@st.cache_data(show_spinner=False)
def video_metadata(path: str, modified: int) -> dict:
    try:
        import cv2

        video = cv2.VideoCapture(path)
        if not video.isOpened():
            return {}
        fps = float(video.get(cv2.CAP_PROP_FPS))
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        result = {"fps": fps, "duration": frames / fps if fps > 0 else 0,
                  "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        video.release()
        return result
    except (ImportError, OSError):
        return {}


def track_ids(summary: dict, observations: pd.DataFrame) -> list[int]:
    values = summary.get("track_ids", [])
    if "track_id" in observations:
        values = observations["track_id"].dropna().unique().tolist()
    return sorted({int(number(value, -1)) for value in values if number(value, -1) >= 0})


def short_time(value: Any) -> str:
    seconds = max(0, number(value))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def render_catalog(catalog: dict, runs: list[tuple[Path, dict]]) -> None:
    clips = catalog.get("clips", [])
    if not clips:
        return
    st.subheader("The shot library")
    st.caption("Each card is a continuous camera shot. Open a prepared result or process its source interval.")
    ready = {summary.get("clip_id", folder.name): folder for folder, summary in runs}
    for offset in range(0, len(clips), 3):
        columns = st.columns(3)
        for column, clip in zip(columns, clips[offset:offset + 3]):
            with column, st.container(border=True):
                thumbnail = project_path(clip.get("thumbnail", ""))
                if thumbnail and thumbnail.is_file():
                    st.image(str(thumbnail), width="stretch")
                st.markdown(f"**{clip.get('label', clip.get('id', 'Continuous shot'))}**")
                start, end = number(clip.get("start_seconds")), number(clip.get("end_seconds"))
                st.caption(f"{short_time(start)} → {short_time(end)} · {end - start:.1f}s · {clip.get('condition', 'Unclassified')}")
                st.caption(f"{str(clip.get('split', 'Unassigned')).title()} split · Run group {clip.get('run_group', 'pending')}")
                clip_id = str(clip.get("id", ""))
                if clip_id in ready:
                    if st.button("Open tracking result", key=f"open_{clip_id}", width="stretch"):
                        st.session_state["selected_run"] = str(ready[clip_id])
                        st.rerun()
                else:
                    st.caption("Ready for processing")


def render_downloads(run_dir: Path, summary: dict) -> None:
    candidates = [
        ("Annotated replay", run_file(run_dir, summary, "video", "annotated.mp4"), "video/mp4"),
        ("Observations CSV", run_file(run_dir, summary, "observations", "observations.csv"), "text/csv"),
        ("Frame metrics CSV", run_file(run_dir, summary, "metrics", "frame_metrics.csv"), "text/csv"),
        ("Run summary JSON", run_dir / "summary.json", "application/json"),
    ]
    for column, (label, path, mime) in zip(st.columns(4), candidates):
        if path and path.is_file():
            with path.open("rb") as handle:
                column.download_button(label, handle, file_name=f"{run_dir.name}_{path.name}", mime=mime,
                                       key=f"download_{run_dir}_{label}", width="stretch")
    for report_name in ("report.md", "report.json", "report.html"):
        report = project_path(run_dir / report_name)
        if report and report.is_file():
            with report.open("rb") as handle:
                st.download_button("Download run report", handle, file_name=f"{run_dir.name}_{report_name}", key=f"report_{run_dir}_{report_name}")


def render_charts(metrics: pd.DataFrame) -> None:
    if metrics.empty or "clip_time" not in metrics:
        st.info("Frame metrics will appear after the clip is processed.")
        return
    data = metrics.copy()
    data["clip_time"] = pd.to_numeric(data["clip_time"], errors="coerce")
    for name in ("lead_observed", "chase_observed", "pair_observed"):
        data[name] = truthy(data[name]) if name in data else False
    separation = pd.to_numeric(data.get("separation_proxy", pd.Series(index=data.index, dtype=float)), errors="coerce")
    data["separation_proxy"] = separation.where(data["pair_observed"])
    valid = data["pair_observed"] & data["separation_proxy"].notna()
    shot_change = data["shot_index"].ne(data["shot_index"].shift()).fillna(False) if "shot_index" in data else pd.Series(False, index=data.index)
    data["segment"] = ((~valid) | shot_change).cumsum()
    st.markdown("**Image separation**")
    st.caption("Detected center separation ÷ mean detected box width. This view dependent image proxy is not physical distance.")
    if valid.any():
        chart = alt.Chart(data.loc[valid]).mark_line(color=ACCENT, strokeWidth=2.5, point=alt.OverlayMarkDef(size=13)).encode(
            x=alt.X("clip_time:Q", title="Time within clip (seconds)"),
            y=alt.Y("separation_proxy:Q", title="Separation / mean box width", scale=alt.Scale(zero=True)),
            detail="segment:N",
            tooltip=[alt.Tooltip("clip_time:Q", title="Clip time", format=".2f"), alt.Tooltip("separation_proxy:Q", title="Image separation", format=".3f")],
        ).properties(height=230).configure_axis(gridColor="#253139", labelColor="#adbcc7", titleColor="#adbcc7").configure_view(stroke=None)
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No valid pair measurements yet. Assign both roles and check whether the two cars were observed together.")
    st.markdown("**Observation availability**")
    availability = data[["clip_time", "lead_observed", "chase_observed"]].melt("clip_time", var_name="role", value_name="observed")
    availability["role"] = availability["role"].map({"lead_observed": "Lead", "chase_observed": "Chase"})
    availability["status"] = availability["observed"].map({True: "Observed", False: "Missing"})
    chart = alt.Chart(availability).mark_tick(thickness=7, size=28).encode(
        x=alt.X("clip_time:Q", title="Time within clip (seconds)"),
        y=alt.Y("role:N", title=None, sort=["Lead", "Chase"]),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Observed", "Missing"], range=[ACCENT, "#ff705f"]), title=None),
        tooltip=[alt.Tooltip("clip_time:Q", title="Clip time", format=".2f"), "role:N", "status:N"],
    ).properties(height=90).configure_axis(gridColor="#253139", labelColor="#adbcc7", titleColor="#adbcc7").configure_view(stroke=None)
    st.altair_chart(chart, width="stretch")
    st.caption("Red marks indicate missing observations. Separation is withheld when either selected car is missing; charts do not fill hidden positions.")


def render_roles(run_dir: Path, summary: dict, observations: pd.DataFrame) -> None:
    ids = track_ids(summary, observations)
    if len(ids) < 2:
        st.warning("Fewer than two tracked IDs are available. Inspect the replay or try another continuous shot.")
        return
    counts = observations.groupby("track_id").size().to_dict() if "track_id" in observations else {}
    labels = {track_id: f"ID {track_id} · {int(counts.get(track_id, 0))} observations" for track_id in ids}
    options = [None] + ids
    lead_id = summary.get("lead_id")
    chase_id = summary.get("chase_id")
    with st.expander("Assign lead and chase", expanded=lead_id is None or chase_id is None):
        st.caption("Watch the numbered replay first, then assign the two participating cars. Roles do not reconnect IDs after a loss or camera cut.")
        with st.form(f"roles_{run_dir}"):
            lead_column, chase_column = st.columns(2)
            lead = lead_column.selectbox("Lead car", options, index=options.index(lead_id) if lead_id in options else 0,
                                         format_func=lambda value: "Choose an ID" if value is None else labels[value])
            chase = chase_column.selectbox("Chase car", options, index=options.index(chase_id) if chase_id in options else 0,
                                           format_func=lambda value: "Choose an ID" if value is None else labels[value])
            submitted = st.form_submit_button("Apply roles and refresh replay", type="primary")
        if submitted:
            if lead is None or chase is None or lead == chase:
                st.error("Choose two different tracked IDs.")
            else:
                try:
                    from driftlens.review import assign_roles

                    with st.spinner("Updating role labels and pair measurements…"):
                        assign_roles(run_dir, int(lead), int(chase))
                    st.session_state["notice"] = "Lead and chase roles saved. The replay and pair measurements were regenerated."
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not assign roles: {error}")


def render_review(runs: list[tuple[Path, dict]]) -> tuple[Path, dict] | None:
    if not runs:
        st.subheader("The first replay is next")
        st.info("Open Process a clip, choose a continuous shot, and run the tracker. The numbered replay will appear here.")
        return None
    run_map = {str(path): (path, summary) for path, summary in runs}
    options = list(run_map)
    if st.session_state.get("selected_run") not in run_map:
        st.session_state["selected_run"] = options[0]
    selected = st.selectbox("Tracking result", options, key="selected_run",
                            format_func=lambda value: f"{run_map[value][1].get('clip_id', Path(value).name)} · {run_map[value][1].get('tracker', 'tracker')} · {number(run_map[value][1].get('duration_seconds')):.1f}s")
    run_dir, summary = run_map[selected]
    observation_file = run_file(run_dir, summary, "observations", "observations.csv")
    metrics_file = run_file(run_dir, summary, "metrics", "frame_metrics.csv")
    observations = read_csv(observation_file) if observation_file else pd.DataFrame()
    metrics = read_csv(metrics_file) if metrics_file else pd.DataFrame()
    roles_ready = summary.get("lead_id") is not None and summary.get("chase_id") is not None
    paired = int(truthy(metrics["pair_observed"]).sum()) if "pair_observed" in metrics else int(number(summary.get("paired_frames")))
    frames = len(metrics) or int(number(summary.get("frame_count")))
    columns = st.columns(4)
    columns[0].metric("Clip duration", f"{number(summary.get('duration_seconds')):.1f}s")
    columns[1].metric("Sampled frames", frames)
    columns[2].metric("Pair observed", f"{100 * paired / frames:.1f}%" if roles_ready and frames else "Assign roles")
    columns[3].metric("Processing speed", f"{number(summary.get('processing_fps')):.2f} fps")
    st.caption(f"{summary.get('tracker', 'Tracker')} · {int(number(summary.get('imgsz')))}px inference · sampled at {number(summary.get('sampled_fps')):g} fps · source {short_time(summary.get('start_seconds'))} to {short_time(summary.get('end_seconds'))}")
    if number(summary.get("shot_count"), 1) > 1:
        st.warning("This interval contains multiple detected shots. Choose a shorter continuous interval before interpreting tandem measurements.")
    reference_check = read_json(run_dir / "evaluation.json")
    sparse_changes = int(number((reference_check.get("tracking") or {}).get("identity_switches")))
    if sparse_changes:
        st.warning(f"Sparse visual reference checks found {sparse_changes} identity change(s) in this shot. Inspect the replay before accepting a lead or chase assignment.")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Annotated replay")
        video = run_file(run_dir, summary, "video", "annotated.mp4")
        if video:
            st.video(str(video))
        else:
            st.warning("The annotated video is missing from this result.")
        render_roles(run_dir, summary, observations)
        if summary.get("role_assignment_method"):
            st.caption("Initial roles were visually reviewed by the AI assistant without human expert validation. Check the participating cars during playback.")
        st.caption("Detector confidence describes each box prediction. It is not a probability that the car identity is correct. Smoke and overlap can cause lost observations or new IDs.")
    with right:
        if roles_ready:
            render_charts(metrics)
        else:
            st.subheader("Identify the tandem pair")
            st.info("Cars begin with numbered IDs. Assign lead and chase beside the replay to generate the pair chart.")
        with st.expander("Tracked IDs and confidence"):
            if {"track_id", "confidence"}.issubset(observations.columns) and not observations.empty:
                table = observations.groupby("track_id").agg(observations=("confidence", "size"), mean_confidence=("confidence", "mean")).reset_index()
                st.dataframe(table, hide_index=True, width="stretch", column_config={"mean_confidence": st.column_config.NumberColumn("Mean detector confidence", format="%.3f")})
            else:
                st.caption("No tracked observations saved for this clip.")
    st.subheader("Take the evidence with you")
    render_downloads(run_dir, summary)
    with st.expander("Inspect raw observations"):
        st.dataframe(observations, hide_index=True, width="stretch")
    return run_dir, summary


def render_process(catalog: dict, sources: list[Path]) -> None:
    st.subheader("Process a continuous shot")
    st.caption("Choose a short interval with both cars visible near its beginning. Keep camera cuts outside the interval.")
    if not sources:
        st.warning("No source video found. Place an MP4 inside the DriftLens folder, then refresh this page.")
        return
    source = st.selectbox("Local source video", sources, format_func=lambda value: str(value.relative_to(ROOT)))
    metadata = video_metadata(str(source), source.stat().st_mtime_ns)
    duration = number(metadata.get("duration"))
    if duration <= 0:
        st.error("This video could not be read. Check the file plays locally and the project environment is installed.")
        return
    st.caption(f"{metadata.get('width', 0)} × {metadata.get('height', 0)} · {number(metadata.get('fps')):.2f} source fps · {short_time(duration)} long")
    catalog_source = project_path((catalog.get("source") or {}).get("path", "longbeach2024_action.mp4"))
    presets = [clip for clip in catalog.get("clips", []) if source == catalog_source]
    choices = {str(clip.get("id")): clip for clip in presets}
    preset_id = st.selectbox("Shot interval", ["custom"] + list(choices),
                             index=1 if choices else 0,
                             format_func=lambda value: "Custom interval" if value == "custom" else str(choices[value].get("label", value)))
    preset = choices.get(preset_id, {})
    start_default = max(0.0, min(duration - 0.1, number(preset.get("start_seconds"))))
    end_default = min(duration, max(start_default + 0.1, number(preset.get("end_seconds"), start_default + 10)))
    with st.form("process_clip"):
        first, second, third = st.columns(3)
        start = first.number_input("Start seconds", min_value=0.0, max_value=duration, value=start_default, step=0.1, key=f"start_{source.name}_{preset_id}")
        end = second.number_input("End seconds", min_value=0.0, max_value=duration, value=end_default, step=0.1, key=f"end_{source.name}_{preset_id}")
        tracker = third.selectbox("Tracker", ["bytetrack", "botsort"], format_func=lambda value: {"bytetrack": "ByteTrack baseline", "botsort": "BoT SORT comparison"}[value])
        first, second, third = st.columns(3)
        imgsz = first.selectbox("Inference image size", [320, 416, 640], index=1)
        target_fps = second.selectbox("Sample frames per second", [5, 10, 15], index=1)
        name_default = str(preset.get("id", "custom_shot"))
        run_name = third.text_input("Result name", value=name_default, key=f"name_{preset_id}")
        replace = st.checkbox("Replace an existing result with this name and tracker")
        submitted = st.form_submit_button("Process clip locally", type="primary", width="stretch")
    st.caption("Start with 5 to 20 seconds and 416px on this computer. Processing is offline using the free model weights installed during setup.")
    if not submitted:
        return
    if end <= start or end - start > 60:
        st.error("Choose an interval longer than zero and no longer than 60 seconds. A short continuous shot gives the clearest first result.")
        return
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", run_name.strip()).strip("_")[:80]
    if not slug:
        st.error("Enter a result name with letters or numbers.")
        return
    output_dir = project_path(RUNS / slug)
    if output_dir is None:
        st.error("The result path is outside this project.")
        return
    existing = read_json(output_dir / "summary.json")
    if existing and existing.get("tracker") != tracker:
        if abs(number(existing.get("start_seconds")) - start) > .01 or abs(number(existing.get("end_seconds")) - end) > .01 or project_path(existing.get("source_path", "")) != source:
            st.error("Use a new result name for a different source interval. Comparisons must use the same source frames.")
            return
        output_dir = output_dir / "comparisons" / tracker
    if (output_dir / "summary.json").exists() and not replace:
        st.error("This result already exists. Choose a new name or enable replacement in the form.")
        return
    progress = st.progress(0, text="Starting local processing…")

    def update_progress(*args: Any, **kwargs: Any) -> None:
        value = args[0] if args else kwargs.get("progress", 0)
        message = args[1] if len(args) > 1 else kwargs.get("message", "Processing frames…")
        if isinstance(value, dict):
            message = value.get("message", message)
            value = value.get("progress", number(value.get("completed")) / max(1, number(value.get("total"), 1)))
        progress.progress(min(1.0, max(0.0, number(value))), text=str(message))

    try:
        from driftlens.pipeline import analyze_video

        result = analyze_video(source=source, output_dir=output_dir, start_seconds=float(start), end_seconds=float(end),
                               tracker=tracker, imgsz=int(imgsz), target_fps=float(target_fps), progress_callback=update_progress)
        progress.progress(1.0, text="Tracking result saved")
        parent_run = output_dir.parent.parent if output_dir.parent.name == "comparisons" else output_dir
        st.session_state["selected_run"] = str(parent_run)
        st.session_state["notice"] = f"Processed {number(result.get('frame_count')):g} sampled frames. Open Run review to inspect the replay and assign lead and chase."
        st.rerun()
    except Exception as error:
        progress.empty()
        st.error(f"Processing failed: {error}")
        st.caption("The local source video is preserved. Inspect the error before retrying.")


def comparison_rows(run_dir: Path, summary: dict) -> list[dict]:
    results = [(run_dir, summary)]
    comparison_dir = run_dir / "comparisons"
    if comparison_dir.exists():
        for path in sorted(comparison_dir.iterdir()):
            safe = project_path(path)
            if safe and safe.is_dir():
                candidate = read_json(safe / "summary.json")
                if candidate and candidate.get("status", "complete") == "complete":
                    results.append((safe, candidate))
    rows = []
    for folder, result in results:
        observations_path = run_file(folder, result, "observations", "observations.csv")
        observations = read_csv(observations_path) if observations_path else pd.DataFrame()
        roles_ready = result.get("lead_id") is not None and result.get("chase_id") is not None
        rows.append({"Tracker": result.get("tracker", folder.name), "Inference px": int(number(result.get("imgsz"))),
                     "Sample fps": number(result.get("sampled_fps")), "Sampled frames": int(number(result.get("frame_count"))),
                     "Tracked IDs": len(track_ids(result, observations)), "Box observations": len(observations),
                     "Pair availability %": number(result.get("pair_coverage")) * 100 if roles_ready else None,
                     "Processing fps": number(result.get("processing_fps")), "Processing seconds": number(result.get("processing_seconds"))})
    return rows


def render_evidence(catalog: dict, runs: list[tuple[Path, dict]], selected: tuple[Path, dict] | None) -> None:
    st.subheader("Measured evidence, visible limitations")
    st.caption("Pair availability measures whether both selected IDs produced observations. It does not establish detection accuracy or correct identity.")
    if selected:
        run_dir, summary = selected
        st.markdown("**Tracker comparison for the selected shot**")
        rows = comparison_rows(run_dir, summary)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(rows) == 1:
            st.info("Process the same source interval with the other tracker and the same result name to add a comparison.")
        st.caption("Trackers can assign different ID numbers. Review and assign roles in each result before comparing pair availability. ID count alone is not an identity switch count.")
        if len(rows) > 1:
            compare_paths = [path for path in (run_dir / "comparisons").iterdir() if path.is_dir() and (path / "summary.json").is_file()]
            for path in compare_paths:
                safe = project_path(path)
                if safe is None:
                    continue
                comparison_summary = read_json(safe / "summary.json")
                with st.expander(f"Inspect {comparison_summary.get('tracker', path.name)} comparison"):
                    comparison_video = run_file(safe, comparison_summary, "video", "annotated.mp4")
                    if comparison_video:
                        st.video(str(comparison_video))
                    observations_path = run_file(safe, comparison_summary, "observations", "observations.csv")
                    render_roles(safe, comparison_summary, read_csv(observations_path) if observations_path else pd.DataFrame())
                    render_downloads(safe, comparison_summary)
    st.markdown("**Dataset manifest**")
    clips = catalog.get("clips", [])
    if clips:
        manifest = [{"Shot": clip.get("id"), "Condition": clip.get("condition"), "Split": clip.get("split"),
                     "Run group": clip.get("run_group"), "Start seconds": clip.get("start_seconds"), "End seconds": clip.get("end_seconds")} for clip in clips]
        st.dataframe(pd.DataFrame(manifest), hide_index=True, width="stretch")
    else:
        st.caption("No shot manifest saved yet.")
    st.markdown("**Reference annotation evaluation**")
    evaluation = ROOT / "outputs" / "evaluation_report.json"
    if not evaluation.is_file():
        evaluation = ROOT / "outputs" / "evaluation" / "evaluation.json"
    if evaluation.is_file():
        saved = read_json(evaluation)
        if saved.get("summary"):
            st.markdown("**Saved evaluation summary**")
            summary_record = saved["summary"]
            confirmed = summary_record.get("confirmed_tracks") or {}
            if confirmed:
                detector = summary_record.get("raw_detector") or {}
                scores = st.columns(3)
                scores[0].metric("Detector precision", f"{100 * number(detector.get('precision')):.1f}%")
                scores[1].metric("Detector recall", f"{100 * number(detector.get('recall')):.1f}%")
                scores[2].metric("Reference test frames", int(number(summary_record.get("labelled_test_frames"))))
                comparison = [{"Tracker": name, "Visible reference coverage %": 100 * number(value.get("visible_reference_coverage")), "Sparse ID changes": value.get("sparse_identity_switches"), "Comparable transitions": value.get("comparable_identity_transitions"), "Processing fps": value.get("processing_fps")} for name, value in confirmed.items()]
                st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
                st.caption("Small diagnostic set using assistant drafted visual references without human expert review. Coverage does not prove uninterrupted identity correctness. Zero sampled changes can accompany missing cars.")
            else:
                st.json(summary_record)
        evaluated_rows = []
        for clip in saved.get("clips", []):
            for tracker_name, values in (clip.get("trackers") or {}).items():
                row = {"Shot": clip.get("clip_id"), "Condition": clip.get("condition"), "Split": clip.get("split"), "Tracker": tracker_name}
                if isinstance(values, dict):
                    row.update({"Visible reference coverage %": 100 * number(values.get("visible_reference_coverage")), "Sparse ID changes": values.get("sparse_identity_switches"), "Comparable transitions": values.get("comparable_identity_transitions"), "Processing fps": values.get("processing_fps")})
                evaluated_rows.append(row)
        if evaluated_rows:
            st.dataframe(pd.DataFrame(evaluated_rows), hide_index=True, width="stretch")
        if saved.get("annotation_method"):
            method = saved["annotation_method"]
            st.caption(f"Annotation method: {method if isinstance(method, str) else json.dumps(method, ensure_ascii=False)}")
        with st.expander("Inspect the complete evaluation record"):
            st.json(saved)
        with evaluation.open("rb") as handle:
            st.download_button("Download saved evaluation", handle, file_name=evaluation.name, mime="application/json")
    else:
        st.info("No reference annotation evaluation is saved yet. No accuracy scores are inferred from detector confidence.")
    st.caption("Evaluation depends on the visible reference boxes and identities. Check the saved annotation method and its review status, keep entire runs in one split, and tag smoke, overlap and complete loss of visibility.")
    results_document = ROOT / "docs" / "RESULTS.md"
    if results_document.is_file():
        with results_document.open("rb") as handle:
            st.download_button("Read the results report · docs/RESULTS.md", handle, file_name="RESULTS.md", mime="text/markdown")
        with st.expander("Results report"):
            st.markdown(results_document.read_text(encoding="utf-8-sig"))
    if runs:
        with st.expander("All measured processing results"):
            measured = []
            for folder, summary in runs:
                row = comparison_rows(folder, summary)[0]
                row["Shot"] = summary.get("clip_id", folder.name)
                measured.append(row)
            st.dataframe(pd.DataFrame(measured), hide_index=True, width="stretch")


catalog = read_json(ROOT / "data" / "clip_catalog.json")
runs = discover_runs()
sources = discover_sources(catalog)
source_info = catalog.get("source") or {}

with st.sidebar:
    st.markdown('<div class="brand">DRIFT<span>LENS</span></div>', unsafe_allow_html=True)
    st.caption("TANDEM TRACKING & RUN REVIEW")
    st.divider()
    st.markdown("**Long Beach · 2024**")
    st.caption(source_info.get("title", "Formula Drift Top 16 · ALL ACTION"))
    if source_info.get("url", "").startswith(("https://", "http://")):
        st.link_button("View footage source", source_info["url"], width="stretch")
    st.divider()
    st.markdown("**Review workflow**")
    st.markdown("1. Choose a continuous shot\n2. Inspect the numbered replay\n3. Assign lead and chase\n4. Review missing observations\n5. Export measured evidence")
    st.divider()
    st.markdown('<div class="side-note">Local footage. Free tools.<br>Image measurements only.<br>No official judging scores.</div>', unsafe_allow_html=True)
    st.caption(f"Project folder: {ROOT}")
    if st.button("Refresh local results", width="stretch"):
        st.rerun()

st.markdown('<div class="eyebrow">COMPUTER VISION / FORMULA DRIFT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Two cars.<br>One continuous story.</h1><p>Follow the tandem pair through real footage. Inspect every detected box, review the moments smoke hides a car, and measure what the camera actually shows.</p><span class="pill">OBSERVED TRACKS</span><span class="pill">NUMBERED REPLAY</span><span class="pill">EXPORTABLE EVIDENCE</span></div>', unsafe_allow_html=True)
stats = st.columns(3)
stats[0].metric("Shots in the library", len(catalog.get("clips", [])))
stats[1].metric("Prepared tracking results", len(runs))
stats[2].metric("Local video sources", len(sources))
if notice := st.session_state.pop("notice", None):
    st.success(notice)
with st.expander(f"Browse the shot library · {len(catalog.get('clips', []))} prepared intervals"):
    render_catalog(catalog, runs)
review_tab, process_tab, evidence_tab = st.tabs(["Run review", "Process a clip", "Evidence"])
with review_tab:
    selected_run = render_review(runs)
with process_tab:
    render_process(catalog, sources)
with evidence_tab:
    render_evidence(catalog, runs, selected_run)
st.markdown('<div class="footer">DriftLens · An independent computer vision portfolio project. Raw footage stays local. Measurements describe the image and do not supply vehicle telemetry.</div>', unsafe_allow_html=True)
