#!/usr/bin/env python3
"""
Thai LJS Recorder (Gradio)
- Records one utterance at a time from your mic
- Saves WAV into ./wavs/uttXXXX.wav (mono, 22050 Hz)
- Appends a row to ./metadata.csv as: uttXXXX|<thai text>
Usage:
  pip install gradio soundfile numpy scipy librosa
  python thai_ljs_recorder.py
Then open the local URL in your browser.
"""
import os, csv, time, io
import numpy as np
import gradio as gr
import soundfile as sf
import librosa

DATA_DIR = os.environ.get("DATA_DIR", ".")
WAV_DIR = os.path.join(DATA_DIR, "wavs")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
os.makedirs(WAV_DIR, exist_ok=True)

SR = 22050


def _next_id():
    # find next integer index based on existing files or metadata rows
    i = 1
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    utt = line.split("|", 1)[0].strip()
                    if utt.startswith("utt"):
                        try:
                            i = max(i, int(utt[3:]) + 1)
                        except:
                            pass
    else:
        # also scan files
        for fn in os.listdir(WAV_DIR):
            if fn.startswith("utt") and fn.endswith(".wav"):
                try:
                    i = max(i, int(fn[3:-4]) + 1)
                except:
                    pass
    return i


def get_last_status():
    """Return markdown showing last row and total count."""
    if not os.path.exists(META_PATH):
        return "ยังไม่มี metadata.csv"
    with open(META_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return "ยังไม่มีรายการใน metadata.csv"
    last = lines[-1]
    total = len(lines)
    if "|" in last:
        utt, text = last.split("|", 1)
        return f"**ไฟล์ล่าสุด:** `{utt}`   \n**ข้อความล่าสุด:** {text}    \n**ทั้งหมด:** {total} แถว"
    return f"**ไฟล์ล่าสุด (raw):** `{last}`  \n**ทั้งหมด:** {total} แถว"


# ---------- Table helpers ----------
def _read_table_rows():
    """Read metadata.csv -> rows [[filename.wav, text, '▶︎ เล่น'], ...]"""
    rows = []
    if not os.path.exists(META_PATH):
        return rows
    with open(META_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or "|" not in ln:
                continue
            utt, text = ln.split("|", 1)
            filename = f"{utt}.wav"
            rows.append([filename, text, "▶︎ เล่น"])
    return rows


def load_table():
    """Return DataFrame value for table tab."""
    return gr.update(value=_read_table_rows())


def _rows_from_table(table):
    """Convert gr.Dataframe value to list-of-lists, robust to pandas.DataFrame."""
    if table is None:
        return []
    # pandas.DataFrame -> list
    try:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            return table.values.tolist()
    except Exception:
        pass
    # dict format (rare)
    if isinstance(table, dict) and "data" in table:
        return table["data"]
    # already list-of-lists
    return table


def play_from_table(evt: gr.SelectData, table):
    """Play audio for the selected row."""
    # evt.index is (row, col)
    idx = None
    if evt is not None and getattr(evt, "index", None) is not None:
        if isinstance(evt.index, (list, tuple)):
            idx = evt.index[0]
        else:
            idx = int(evt.index)
    rows = _rows_from_table(table)
    if idx is None or idx < 0 or idx >= len(rows):
        return gr.update()  # no change
    filename = rows[idx][0]  # first column is filename 'uttXXXX.wav'
    path = os.path.join(WAV_DIR, filename)
    return path if os.path.exists(path) else gr.update()


# -----------------------------------


def save_sample(audio, text, manual_id, normalize=True, trim=True):
    # ถ้ายังไม่ใส่ข้อความ
    if not text or not text.strip():
        # ไม่เปลี่ยนค่า next_id_out / manual_id / last_row / table
        return (
            gr.update(value=""),
            "⚠️ ใส่ข้อความก่อน",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # audio is (sr, data) when "microphone=True" in Gradio
    if audio is None or audio[1] is None:
        return (
            text,
            "⚠️ ยังไม่มีเสียง (กดอัด/Allow mic)",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    sr, data = audio
    y = np.array(data, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)  # mono
    # resample
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    # optional trim
    if trim:
        yt, _ = librosa.effects.trim(y, top_db=30)
        if len(yt) > int(0.3 * SR):  # keep >300ms
            y = yt
    # optional normalize
    if normalize:
        peak = np.max(np.abs(y)) + 1e-9
        y = y / peak * 0.98

    # pick id
    idx = int(manual_id) if manual_id else _next_id()
    utt_id = f"utt{idx:04d}"
    wav_path = os.path.join(WAV_DIR, f"{utt_id}.wav")
    sf.write(wav_path, y, SR, subtype="PCM_16")

    # append metadata
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(f"{utt_id}|{text.strip()}\n")

    next_hint = idx + 1
    # ล้างช่องข้อความ, แสดงสถานะ, ตั้งค่า Id ถัดไป (ทั้งช่องแนะนำและช่องกรอก),
    # อัปเดตบรรทัดล่าสุด, และอัปเดตตารางในแท็บ CSV
    return (
        "",
        f"✅ Saved {utt_id}.wav and appended to metadata.csv",
        str(next_hint),
        str(next_hint),
        get_last_status(),
        load_table(),
    )


def undo_last():
    # remove last row and file
    if not os.path.exists(META_PATH):
        return "ไม่มี metadata.csv", get_last_status(), load_table()
    lines = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return "ไม่มีรายการให้ลบ", get_last_status(), load_table()
    last = lines[-1].strip()
    if "|" in last:
        utt_id = last.split("|", 1)[0]
        wav_path = os.path.join(WAV_DIR, f"{utt_id}.wav")
        if os.path.exists(wav_path):
            os.remove(wav_path)
    with open(META_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])
    return f"↩️ ลบ {last} แล้ว", get_last_status(), load_table()


# ---------- prompt loader + navigator (next/prev) ----------
def load_prompts(file_input):
    """Read prompts.txt -> return list, reset idx=0, set textbox to first."""
    if not file_input:
        return [], 0, "", "อัปโหลดไฟล์ .txt ที่มี 1 บรรทัด/ประโยค", "0/0"

    path = getattr(file_input, "name", None) or str(file_input)
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            content = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except Exception as e:
        return [], 0, "", f"อ่านไฟล์ไม่ได้: {e}", "0/0"

    first = content[0] if content else ""
    prog = f"1/{len(content)}" if content else "0/0"
    return content, 0, first, f"โหลด {len(content)} บรรทัด", prog


def _nav(prompts, idx, step):
    """Move index by step (+1 next, -1 prev)."""
    n = len(prompts)
    if n == 0:
        return 0, "", "0/0"
    new_idx = int(idx) + int(step)
    new_idx = max(0, min(n - 1, new_idx))
    return new_idx, prompts[new_idx], f"{new_idx + 1}/{n}"


def go_next(prompts, idx):
    return _nav(prompts, idx, +1)


def go_prev(prompts, idx):
    return _nav(prompts, idx, -1)


# ----------------------------------------------------------------


with gr.Blocks() as demo:
    gr.Markdown("# Thai LJS Recorder")

    with gr.Tabs():
        # ---------------- Tab 1: Recorder ----------------
        with gr.Tab("บันทึก"):
            gr.Markdown("บันทึกทีละประโยค → ได้ไฟล์ WAV และเพิ่มแถวใน metadata.csv อัตโนมัติ")
            with gr.Row():
                mic = gr.Audio(sources=["microphone"], type="numpy")
                with gr.Column():
                    # Prev/Next + progress
                    prompts_state = gr.State([])  # list[str]
                    idx_state = gr.State(0)  # current index
                    with gr.Row():
                        prev_btn = gr.Button("ก่อนหน้า")
                        next_btn = gr.Button("ถัดไป")
                        progress = gr.Markdown("0/0")  # shows "i/N"
                    # Text + controls
                    txt = gr.Textbox(label="ข้อความ (ไทย)")
                    with gr.Row():
                        manual_id = gr.Textbox(
                            label="เลขไอดีถัดไป (ว่าง = อัตโนมัติ)", value=str(_next_id())
                        )
                        next_id_out = gr.Textbox(
                            label="Id ถัดไป (แนะนำ)", interactive=False
                        )
                    with gr.Row():
                        normalize = gr.Checkbox(label="Normalize", value=True)
                        trim = gr.Checkbox(label="Auto-trim", value=True)
                    btn = gr.Button("บันทึก")
                    status = gr.Markdown()
                    undo = gr.Button("↩️ Undo รายการล่าสุด")
                    status2 = gr.Markdown()
                    # Last row display + refresh
                    last_row_md = gr.Markdown("ล่าสุด: -")
                    refresh_last = gr.Button("รีเฟรชรายการล่าสุด")

            with gr.Row():
                prompts_txt = gr.File(
                    label="อัปโหลด prompts.txt (1 บรรทัด/ประโยค)",
                    file_types=[".txt"],
                    type="filepath",
                )
                load_status = gr.Markdown()

        # ---------------- Tab 2: CSV Table ----------------
        with gr.Tab("ตาราง CSV"):
            gr.Markdown("ตารางจาก `metadata.csv`")
            table = gr.Dataframe(
                headers=["ชื่อไฟล์", "ข้อความ", "เล่นไฟล์เสียงนี้"],
                value=[],
                interactive=False,
                row_count=(0, "dynamic"),
                col_count=(3, "fixed"),
                wrap=True,
            )
            with gr.Row():
                table_refresh = gr.Button("รีเฟรชตาราง")
                audio_player = gr.Audio(label="ตัวอย่างเสียง", autoplay=False)

    # ----- wiring -----

    # init last-row & table on load
    demo.load(
        lambda: (get_last_status(), _read_table_rows()),
        inputs=None,
        outputs=[last_row_md, table],
    )

    # refresh buttons
    refresh_last.click(lambda: get_last_status(), inputs=None, outputs=[last_row_md])
    table_refresh.click(load_table, inputs=None, outputs=[table])

    # upload prompts -> store list, reset idx=0, put first into textbox, update status+progress
    prompts_txt.upload(
        load_prompts,
        inputs=[prompts_txt],
        outputs=[prompts_state, idx_state, txt, load_status, progress],
    )

    # navigate
    next_btn.click(
        go_next, inputs=[prompts_state, idx_state], outputs=[idx_state, txt, progress]
    )
    prev_btn.click(
        go_prev, inputs=[prompts_state, idx_state], outputs=[idx_state, txt, progress]
    )

    # save -> also update table
    btn.click(
        save_sample,
        inputs=[mic, txt, manual_id, normalize, trim],
        outputs=[txt, status, next_id_out, manual_id, last_row_md, table],
    )

    # undo -> also update table
    undo.click(lambda: undo_last(), inputs=None, outputs=[status2, last_row_md, table])

    # click any cell -> play that row's audio
    table.select(play_from_table, inputs=[table], outputs=[audio_player])

if __name__ == "__main__":
    demo.launch()
