"""
module2_mod.py — реальное время + прослушивание.

Совместимо с прошивкой STM32 (уже залита):
  RX: 0xAA 0x55 + 8×uint8   (10 байт)
  TX: 0xAA 0x55 + A + B     (4 байта)
  A = samples[7]
  B = (sum(samples)+4) >> 3

Возможности:
  - живой график с побайтовой сверкой ответа МК и эталона на ПК
  - кнопки ▶ прослушивания: слева Вход, справа A и B
  - SPACE — пауза, 1/2/3 — прослушать Вход/A/B

Требует: pip install pyserial numpy matplotlib scipy sounddevice
"""

import time
import argparse
import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque

try:
    import sounddevice as sd
    from scipy.signal import resample as scipy_resample
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[!] sounddevice/scipy не установлены — плеер отключён.")
    print("    pip install sounddevice scipy")


# ================= НАСТРОЙКИ =================
COM_PORT   = "COM6"
BAUD_RATE  = 115200

FS_IN         = 2000.0
DECIM         = 8
FS_OUT        = FS_IN / DECIM
PACKET_PERIOD = DECIM / FS_IN

WINDOW_SIZE = 30                     # окно на экране

# --- аудио ---
AUDIO_HISTORY_SEC = 8.0              # сколько секунд истории держим
AUDIO_FS          = 44100
AUDIO_GAIN        = 0.5
FREQ_SHIFT        = 1.0              # 60 Гц → 240 Гц (слышно на любых колонках)

# --- сигнал ---
AMPLITUDE = 0.6
F_SIGNAL  = 500.0
SIGMA     = 1.0

HEADER       = bytes([0xAA, 0x55])
RESPONSE_LEN = 2


# ================= UART =================
def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.02, write_timeout=0.2)
    except serial.SerialException as e:
        raise SystemExit(f"Ошибка подключения: {e}")
    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("UART подключен:", port)
    return ser


def read_response(ser):
    state = 0
    while True:
        b = ser.read(1)
        if not b:
            return None
        v = b[0]
        if state == 0:
            if v == 0xAA:
                state = 1
        else:
            if v == 0x55:
                payload = ser.read(RESPONSE_LEN)
                if len(payload) != RESPONSE_LEN:
                    return None
                return payload[0], payload[1]
            elif v == 0xAA:
                state = 1
            else:
                state = 0


def generate_and_send(ser, sample_index, rng):
    indices = sample_index + np.arange(DECIM)
    t_arr   = indices / FS_IN

    ideal    = AMPLITUDE * np.sin(2 * np.pi * F_SIGNAL * t_arr)
    ideal_8b = np.rint(ideal * 127.0 + 128.0)

    noise = rng.normal(0.0, SIGMA, DECIM)
    noisy = np.clip(np.rint(ideal_8b + noise), 0, 255).astype(np.uint8)

    ref_A = int(noisy[7])
    s     = int(noisy.astype(np.uint16).sum())
    ref_B = (s + 4) >> 3

    packet = HEADER + noisy.tobytes()
    if ser.write(packet) != len(packet):
        return None

    resp = read_response(ser)
    if resp is None:
        return None
    raw_A, raw_B = resp

    return (sample_index + DECIM, int(raw_A), int(raw_B), ref_A, ref_B, noisy)


# ================= AUDIO =================
def prepare_audio(samples_uint8, fs_orig, gain, freq_shift):
    """uint8 0..255 → float32 [-1,+1] на AUDIO_FS. freq_shift ускоряет ×N."""
    x = np.asarray(samples_uint8, dtype=np.float32) - 128.0
    if x.size < 32:
        return None
    new_len = int(x.size * AUDIO_FS / fs_orig / freq_shift)
    if new_len < 32:
        return None
    y = scipy_resample(x, new_len).astype(np.float32)
    y = y / 128.0 * gain
    return np.clip(y, -1.0, 1.0)


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=COM_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--gain", type=float, default=AUDIO_GAIN)
    parser.add_argument("--shift", type=float, default=FREQ_SHIFT,
                        help="множитель частоты: 1=native 60 Hz, 4=240 Hz")
    args = parser.parse_args()

    ser = open_serial(args.port, args.baud)
    rng = np.random.default_rng()

    sample_index = 0
    stats_ok = 0
    stats_bad = 0

    # --- буферы графика ---
    time_input   = deque(maxlen=WINDOW_SIZE * DECIM)
    input_buffer = deque(maxlen=WINDOW_SIZE * DECIM)
    time_A       = deque(maxlen=WINDOW_SIZE)
    time_B       = deque(maxlen=WINDOW_SIZE)
    track_A_buf  = deque(maxlen=WINDOW_SIZE)
    track_B_buf  = deque(maxlen=WINDOW_SIZE)

    # --- буферы аудио ---
    audio_in = deque(maxlen=int(AUDIO_HISTORY_SEC * FS_IN))
    audio_A  = deque(maxlen=int(AUDIO_HISTORY_SEC * FS_OUT))
    audio_B  = deque(maxlen=int(AUDIO_HISTORY_SEC * FS_OUT))

    # --- плеер ---
    played_label = [""]
    played_until = [0.0]

    def do_play(buffer_deque, fs_orig, label):
        if not AUDIO_AVAILABLE:
            print("[play] звук недоступен")
            return
        min_len = int(0.25 * fs_orig)            # нужно хотя бы 250 мс
        if len(buffer_deque) < min_len:
            print(f"[play] {label}: мало данных ({len(buffer_deque)}/{min_len})")
            return
        samples = np.fromiter(buffer_deque, dtype=np.uint8,
                              count=len(buffer_deque))
        y = prepare_audio(samples, fs_orig, args.gain, args.shift)
        if y is None:
            return
        try:
            sd.stop()
            sd.play(y, AUDIO_FS, blocking=False)
        except Exception as e:
            print(f"[play] ошибка: {e}")
            return
        played_label[0] = label
        played_until[0] = time.time() + len(y) / AUDIO_FS
        print(f"[play] {label}: {len(y)/AUDIO_FS:.2f} с, "
              f"{len(samples)} отсч. @ {fs_orig:.0f} Гц → ×{args.shift}")

    # --- графики ---
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.86, top=0.94, bottom=0.08, hspace=0.35)
    fig.canvas.manager.set_window_title(
        "STM32 DSP — SPACE: pause, 1/2/3: play")

    line_in_A, = ax1.plot([], [], color="gray", alpha=0.55, linewidth=1,
                          label="Вход 2000 Гц")
    line_A, = ax1.plot([], [], "o-", color="red", markersize=5, linewidth=1.5,
                       label="Тракт A (250 Гц)")
    ax1.set_title("Тракт A — прореживание", color="#ff5555")
    ax1.set_ylabel("Амплитуда, LSB")
    ax1.set_ylim(-100, 100)
    ax1.axhline(0, color="white", linestyle="--", alpha=0.35)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    line_in_B, = ax2.plot([], [], color="gray", alpha=0.55, linewidth=1,
                          label="Вход 2000 Гц")
    line_B, = ax2.plot([], [], "o-", color="blue", markersize=5, linewidth=1.5,
                       label="Тракт B (250 Гц)")
    ax2.set_title("Тракт B — усреднение", color="#4da6ff")
    ax2.set_ylabel("Амплитуда, LSB")
    ax2.set_xlabel("Время, секунды")
    ax2.set_ylim(-100, 100)
    ax2.axhline(0, color="white", linestyle="--", alpha=0.35)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right")

    status_text = ax1.text(0.015, 0.94, "RUNNING", transform=ax1.transAxes,
                           color="lime", fontsize=12, fontweight="bold", va="top")
    info_text = ax1.text(0.015, 0.82, "", transform=ax1.transAxes,
                         color="white", fontsize=10, va="top", family="monospace")

    # --- кнопки ▶ ---
    if AUDIO_AVAILABLE:
        # Слева — Вход
        ax_btn_in = fig.add_axes([0.02, 0.42, 0.09, 0.16])
        btn_in = Button(ax_btn_in, "▶ Вход\nшумный",
                        color="#333333", hovercolor="#555555")
        btn_in.label.set_fontsize(10)
        btn_in.on_clicked(lambda e: do_play(audio_in, FS_IN, "Вход"))

        # Справа от A
        ax_btn_A = ax1.inset_axes([1.02, 0.30, 0.08, 0.40])
        btn_A = Button(ax_btn_A, "▶\nТракт A",
                       color="#3a1c1c", hovercolor="#5a2c2c")
        btn_A.label.set_fontsize(10)
        btn_A.on_clicked(lambda e: do_play(audio_A, FS_OUT, "Тракт A"))

        # Справа от B
        ax_btn_B = ax2.inset_axes([1.02, 0.30, 0.08, 0.40])
        btn_B = Button(ax_btn_B, "▶\nТракт B",
                       color="#1c2a3a", hovercolor="#2c3a5a")
        btn_B.label.set_fontsize(10)
        btn_B.on_clicked(lambda e: do_play(audio_B, FS_OUT, "Тракт B"))

    # --- пауза и клавиши ---
    paused = [False]
    last_period_ms = [0.0]
    next_packet_time = [time.perf_counter()]

    def toggle_pause():
        paused[0] = not paused[0]
        status_text.set_text("PAUSED" if paused[0] else "RUNNING")
        status_text.set_color("red" if paused[0] else "lime")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            toggle_pause()
        elif event.key == "1":
            do_play(audio_in, FS_IN, "Вход")
        elif event.key == "2":
            do_play(audio_A, FS_OUT, "Тракт A")
        elif event.key == "3":
            do_play(audio_B, FS_OUT, "Тракт B")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def init():
        for ln in (line_in_A, line_in_B, line_A, line_B):
            ln.set_data([], [])
        return line_in_A, line_in_B, line_A, line_B

    def update(frame):
        nonlocal sample_index, stats_ok, stats_bad

        if not paused[0]:
            now = time.perf_counter()
            t_batch = now
            sent = 0
            while now >= next_packet_time[0] and sent < 4:
                res = generate_and_send(ser, sample_index, rng)
                if res is None:
                    stats_bad += 1
                    sample_index += DECIM
                    ser.reset_input_buffer()
                    next_packet_time[0] += PACKET_PERIOD
                    now = time.perf_counter()
                    sent += 1
                    continue

                sample_index, A, B, refA, refB, noisy = res
                if A == refA and B == refB:
                    stats_ok += 1
                else:
                    stats_bad += 1

                base = sample_index - DECIM
                for i in range(DECIM):
                    time_input.append((base + i) / FS_IN)
                    input_buffer.append(int(noisy[i]) - 128)
                    audio_in.append(int(noisy[i]))

                time_A.append((base + 7.0) / FS_IN)
                track_A_buf.append(A - 128)
                audio_A.append(A)

                time_B.append((base + 3.5) / FS_IN)
                track_B_buf.append(B - 128)
                audio_B.append(B)

                next_packet_time[0] += PACKET_PERIOD
                now = time.perf_counter()
                sent += 1

            if sent > 0:
                last_period_ms[0] = (time.perf_counter() - t_batch) * 1000.0 / sent

        if len(time_input) > 0:
            line_in_A.set_data(time_input, input_buffer)
            line_in_B.set_data(time_input, input_buffer)
            line_A.set_data(time_A, track_A_buf)
            line_B.set_data(time_B, track_B_buf)
            right = sample_index / FS_IN
            left  = max(0.0, right - WINDOW_SIZE / FS_OUT)
            ax1.set_xlim(left, max(left + 0.02, right))
            ax2.set_xlim(left, max(left + 0.02, right))

        total = max(stats_ok + stats_bad, 1)
        play_txt = f"  ▶ {played_label[0]}" if time.time() < played_until[0] else ""
        info_text.set_text(
            f"OK={stats_ok:6d}  BAD={stats_bad:4d}  "
            f"Match={100*stats_ok/total:5.2f}%\n"
            f"Пакет: {last_period_ms[0]:5.2f} мс  "
            f"(цель {PACKET_PERIOD*1000:.2f} мс){play_txt}"
        )

        return (line_in_A, line_in_B, line_A, line_B,
                status_text, info_text)

    print("=" * 60)
    print("STM32 DSP REAL-TIME + AUDIO PLAYBACK")
    print("  SPACE — пауза/продолжить")
    print("  1/2/3 — прослушать: Вход / A / B")
    print("  Кнопки: слева Вход, справа A и B")
    print("=" * 60)
    if not AUDIO_AVAILABLE:
        print("[!] звук недоступен, кнопки не отрисованы")
    if args.shift != 1.0:
        print(f"[i] частота сдвинута ×{args.shift}: 60 → "
              f"{60*args.shift:.0f} Гц (для слышимости)")

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  interval=10, blit=False,
                                  cache_frame_data=False)
    try:
        plt.show()
    finally:
        ser.close()
        if AUDIO_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        total = max(stats_ok + stats_bad, 1)
        print(f"Итого: OK={stats_ok}/{total} "
              f"({100*stats_ok/total:.2f}%)  BAD={stats_bad}")
        print("UART отключен")


if __name__ == "__main__":
    main()