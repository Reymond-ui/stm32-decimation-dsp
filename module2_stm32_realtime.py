"""
module2_stm32_realtime.py
Работает с прошивкой:
  RX: 0xAA 0x55 + 8×uint8   (10 байт)
  TX: 0xAA 0x55 + A + B     (4 байта)
  A = samples[7]
  B = (sum(samples)+4) >> 3
"""

import time
import argparse
import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ================= НАСТРОЙКИ =================
COM_PORT   = "COM6"
BAUD_RATE  = 115200

FS_IN         = 2000.0
DECIM         = 8
FS_OUT        = FS_IN / DECIM
PACKET_PERIOD = DECIM / FS_IN        # 4 мс

WINDOW_SIZE = 30                     # ~120 мс на экране

AMPLITUDE = 0.6
F_SIGNAL  = 60.0
SIGMA     = 2.0

HEADER       = bytes([0xAA, 0x55])
RESPONSE_LEN = 2                     # A, B (после заголовка)

# ================= UART =================
def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.02, write_timeout=0.2)
    except serial.SerialException as e:
        raise SystemExit(f"Ошибка подключения: {e}")
    time.sleep(1.5)                   # прошивка печатает баннер
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("UART подключен:", port)
    return ser


def read_response(ser):
    """Ищем заголовок AA 55, читаем A и B. None при тайм-ауте."""
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
    """
    Возвращает (new_index, A, B, ref_A, ref_B, noisy_bytes) или None.
    ref_A/ref_B — эталон, посчитанный на ПК по той же логике, что и в прошивке.
    """
    indices = sample_index + np.arange(DECIM)
    t_arr   = indices / FS_IN

    ideal     = AMPLITUDE * np.sin(2 * np.pi * F_SIGNAL * t_arr)
    ideal_8b  = np.rint(ideal * 127.0 + 128.0)

    noise = rng.normal(0.0, SIGMA, DECIM)
    noisy = np.clip(np.rint(ideal_8b + noise), 0, 255).astype(np.uint8)

    # ----- эталон -----
    ref_A = int(noisy[7])
    s     = int(noisy.astype(np.uint16).sum())
    ref_B = (s + 4) >> 3

    # ----- отправка -----
    packet = HEADER + noisy.tobytes()
    if ser.write(packet) != len(packet):
        return None

    # ----- приём -----
    resp = read_response(ser)
    if resp is None:
        return None
    raw_A, raw_B = resp

    return (sample_index + DECIM, int(raw_A), int(raw_B), ref_A, ref_B, noisy)


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=COM_PORT)
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--no-plot", action="store_true",
                        help="прогон без GUI, только проверка совпадения")
    parser.add_argument("--n", type=int, default=1000,
                        help="сколько пакетов в режиме --no-plot")
    args = parser.parse_args()

    ser = open_serial(args.port, args.baud)
    rng = np.random.default_rng()

    sample_index = 0
    stats_ok = 0
    stats_bad = 0

    # -------- режим только-валидация --------
    if args.no_plot:
        print(f"Проверка {args.n} пакетов без GUI...")
        next_time = time.perf_counter()
        for _ in range(args.n):
            res = generate_and_send(ser, sample_index, rng)
            if res is None:
                stats_bad += 1
                sample_index += DECIM
                ser.reset_input_buffer()
            else:
                sample_index, A, B, refA, refB, _ = res
                if A == refA and B == refB:
                    stats_ok += 1
                else:
                    stats_bad += 1
                    print(f"MISMATCH @ {sample_index:8d}: "
                          f"A={A:3d}/{refA:3d} B={B:3d}/{refB:3d}")
            next_time += PACKET_PERIOD
            dt = next_time - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        total = stats_ok + stats_bad
        print(f"Итог: OK={stats_ok}/{total} ({100*stats_ok/max(total,1):.2f}%)  BAD={stats_bad}")
        ser.close()
        return

    # -------- буферы для графика --------
    time_input   = deque(maxlen=WINDOW_SIZE * DECIM)
    input_buffer = deque(maxlen=WINDOW_SIZE * DECIM)
    time_A       = deque(maxlen=WINDOW_SIZE)
    time_B       = deque(maxlen=WINDOW_SIZE)
    track_A_buf  = deque(maxlen=WINDOW_SIZE)
    track_B_buf  = deque(maxlen=WINDOW_SIZE)

    paused = False
    last_period_ms = 0.0
    next_packet_time = time.perf_counter()

    # -------- графики --------
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title("STM32 DSP — SPACE: Pause/Resume")

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

    def toggle_pause(event):
        nonlocal paused
        if event.key == " ":
            paused = not paused
            status_text.set_text("PAUSED" if paused else "RUNNING")
            status_text.set_color("red" if paused else "lime")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", toggle_pause)

    def init():
        for ln in (line_in_A, line_in_B, line_A, line_B):
            ln.set_data([], [])
        return line_in_A, line_in_B, line_A, line_B

    def update(frame):
        nonlocal sample_index, next_packet_time, last_period_ms
        nonlocal stats_ok, stats_bad, paused

        if not paused:
            now = time.perf_counter()
            t_batch = now
            sent = 0
            while now >= next_packet_time and sent < 4:
                res = generate_and_send(ser, sample_index, rng)

                if res is None:
                    stats_bad += 1
                    sample_index += DECIM
                    ser.reset_input_buffer()
                    next_packet_time += PACKET_PERIOD
                    now = time.perf_counter()
                    sent += 1
                    continue

                sample_index, A, B, refA, refB, noisy = res

                if A == refA and B == refB:
                    stats_ok += 1
                else:
                    stats_bad += 1

                # входные отсчёты (0..255 -> -128..127)
                base = sample_index - DECIM
                for i in range(DECIM):
                    time_input.append((base + i) / FS_IN)
                    input_buffer.append(int(noisy[i]) - 128)

                # выход: A — последний в окне (задержка 7/Fs)
                time_A.append((base + 7.0) / FS_IN)
                track_A_buf.append(A - 128)
                # B — центр окна усреднения (задержка 3.5/Fs)
                time_B.append((base + 3.5) / FS_IN)
                track_B_buf.append(B - 128)

                next_packet_time += PACKET_PERIOD
                now = time.perf_counter()
                sent += 1

            if sent > 0:
                last_period_ms = (time.perf_counter() - t_batch) * 1000.0 / sent

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
        info_text.set_text(
            f"OK={stats_ok:6d}  BAD={stats_bad:4d}  "
            f"Match={100*stats_ok/total:5.2f}%\n"
            f"Пакет: {last_period_ms:5.2f} мс  (цель {PACKET_PERIOD*1000:.2f} мс)"
        )

        return (line_in_A, line_in_B, line_A, line_B, status_text, info_text)

    print("=" * 55)
    print("STM32 DSP REAL-TIME")
    print("  SPACE — пауза/продолжить")
    print("  Закрыть окно — выход")
    print("=" * 55)

    ani = animation.FuncAnimation(fig, update, init_func=init, interval=10,
                                  blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        ser.close()
        total = max(stats_ok + stats_bad, 1)
        print(f"Итого: OK={stats_ok}/{total} ({100*stats_ok/total:.2f}%)  BAD={stats_bad}")
        print("UART отключен")


if __name__ == "__main__":
    main()