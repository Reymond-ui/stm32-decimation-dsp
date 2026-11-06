import numpy as np
import matplotlib.pyplot as plt

# ================= НАСТРОЙКИ =================
FS_IN    = 2000.0
DECIM    = 8
FS_OUT   = FS_IN / DECIM

DURATION = 4.0
F_SIGNAL = 60.0
AMPLITUDE = 0.6
SIGMA    = 64.0
SEED     = 7
N_SHOW   = 30

# ================= ГЕНЕРАЦИЯ СИГНАЛА =================
N_SAMPLES = int(FS_IN * DURATION)
t = np.arange(N_SAMPLES) / FS_IN

# Полезный сигнал в формате unsigned 8-bit (DC = 128)
ideal_signal = AMPLITUDE * np.sin(2 * np.pi * F_SIGNAL * t)
ideal_8bit   = np.rint(ideal_signal * 127.0 + 128.0)

# Аналоговый шум sigma = 2 LSB на входе АЦП
rng   = np.random.default_rng(SEED)
noise = rng.normal(0.0, SIGMA, N_SAMPLES)

# ИСПРАВЛЕНО: np.rint перед astype(uint8) — устраняет усечение к нулю
signal_8bit = np.clip(
    np.rint(ideal_8bit + noise),
    0, 255
).astype(np.uint8)

# ================= ДЕЦИМАЦИЯ =================
n_blocks = len(signal_8bit) // DECIM

# ИСПРАВЛЕНО: uint16 достаточно (max = 8*255 = 2040), uint32 избыточен
blocks = (
    signal_8bit[:n_blocks * DECIM]
    .astype(np.uint16)
    .reshape(-1, DECIM)
)

# Тракт A: каждый восьмой (последний в окне)
track_A = blocks[:, 7].astype(np.uint8)

# Тракт B: сумма восьми, округление (+4) и сдвиг >> 3
sums    = blocks.sum(axis=1, dtype=np.uint16)
track_B = ((sums + 4) >> 3).astype(np.uint8)

# ================= ЗНАКОВЫЙ ДИАПАЗОН =================
track_A_signed = track_A.astype(float) - 128.0
track_B_signed = track_B.astype(float) - 128.0

# ================= ВРЕМЕННЫЕ ОСИ =================
k = np.arange(n_blocks)

# Тракт A: задержка 7/Fs — взят последний отсчёт окна
t_A = (k * DECIM + 7.0) / FS_IN
# Тракт B: задержка 3.5/Fs — центр окна усреднения
t_B = (k * DECIM + 3.5) / FS_IN

# ================= ИДЕАЛЬНЫЕ СИГНАЛЫ =================
ideal_A = AMPLITUDE * 127.0 * np.sin(2 * np.pi * F_SIGNAL * t_A)

# КЧХ усреднителя-8 на частоте 60 Гц (boxcar)
H_60 = abs(
    np.sin(DECIM * np.pi * F_SIGNAL / FS_IN) /
    (DECIM * np.sin(np.pi * F_SIGNAL / FS_IN))
)
ideal_B = AMPLITUDE * 127.0 * H_60 * np.sin(2 * np.pi * F_SIGNAL * t_B)

# ================= МЕТРИКИ =================
def calculate_metrics(signal, ideal):
    error      = signal - ideal
    rms_noise  = np.sqrt(np.mean(error ** 2))
    rms_signal = np.sqrt(np.mean(ideal ** 2))
    snr        = 20 * np.log10(rms_signal / max(rms_noise, 1e-15))
    return rms_noise, snr

rms_A, snr_A = calculate_metrics(track_A_signed, ideal_A)
rms_B, snr_B = calculate_metrics(track_B_signed, ideal_B)

gain_real   = snr_B - snr_A
gain_no_H   = 10 * np.log10(DECIM)                        # 9.03 дБ (идеал)
gain_with_H = 20 * np.log10(H_60 * np.sqrt(DECIM))        # реальный предел
att_dB      = 20 * np.log10(H_60)

# ================= ВЫВОД =================
print("=" * 66)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ DSP")
print("=" * 66)
print(f"Частота входа:            {FS_IN:.0f} Гц")
print(f"Частота выхода:           {FS_OUT:.0f} Гц")
print(f"Полезная частота:         {F_SIGNAL:.0f} Гц")
print(f"Sigma аналог. шума:       {SIGMA:.2f} LSB")
print(f"Ослабление H(60 Гц):      {H_60:.5f}  ({att_dB:+.2f} дБ)")
print("-" * 66)
print(f"Тракт A: RMS = {rms_A:6.3f} LSB,  SNR = {snr_A:6.2f} дБ")
print(f"Тракт B: RMS = {rms_B:6.3f} LSB,  SNR = {snr_B:6.2f} дБ")
print("-" * 66)
print(f"Измеренный выигрыш B/A:   {gain_real:+.2f} дБ")
print(f"Теория без H (идеал):     {gain_no_H:+.2f} дБ  (= 10·log10 8)")
print(f"Теория с H(60):           {gain_with_H:+.2f} дБ  "
      f"(сигнал ослаблен {att_dB:+.2f} дБ)")
print("=" * 66)

# Проверка эквивалентности целочисленной модели и float-эталона
x_signed  = signal_8bit.astype(float) - 128.0
y_B_float = x_signed[:n_blocks * DECIM].reshape(-1, DECIM).mean(axis=1)
diff_B    = np.max(np.abs(track_B_signed - y_B_float))
print(f"max |int_B - float_B| = {diff_B:.3f} LSB  (ожидается ≤ 0.5 LSB)")
print("=" * 66)

# ================= FFT =================
def spectrum_db(signal, fs):
    s = np.asarray(signal, dtype=float)
    s = s - np.mean(s)
    N = len(s)
    window   = np.hanning(N)
    spectrum = np.fft.rfft(s * window)
    amplitude = np.abs(spectrum) / np.sum(window)
    if len(amplitude) > 2:
        amplitude[1:-1] *= 2
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    db   = 20 * np.log10(np.maximum(amplitude, 1e-12))
    return freq, db

freq_A, fft_A = spectrum_db(track_A_signed, FS_OUT)
freq_B, fft_B = spectrum_db(track_B_signed, FS_OUT)

# Теоретическая АЧХ boxcar-8 (для наложения)
f_env = np.linspace(0.0, FS_OUT / 2, 800)
H_env = np.ones_like(f_env)
mask  = f_env > 0.01
H_env[mask] = np.abs(
    np.sin(DECIM * np.pi * f_env[mask] / FS_IN) /
    (DECIM * np.sin(np.pi * f_env[mask] / FS_IN))
)
env_dB = 20 * np.log10(AMPLITUDE * 127.0 * H_env)

# ================= ВИЗУАЛИЗАЦИЯ =================
plt.style.use("dark_background")
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("ЦОС: сравнение двух методов децимации", fontsize=16)

N_IN    = N_SHOW * DECIM
t_plot  = t[:N_IN] * 1000
in_plot = signal_8bit[:N_IN].astype(int) - 128

# --- График 1: тракт A ---
axs[0, 0].plot(t_plot, in_plot, color="gray", alpha=0.55, label="Вход 2000 Гц")
axs[0, 0].plot(t_A[:N_SHOW] * 1000, track_A_signed[:N_SHOW], "o-",
               color="red", markersize=5, label="Тракт A (250 Гц)")
axs[0, 0].set_title("Тракт A — прореживание")
axs[0, 0].set_xlabel("Время, мс")
axs[0, 0].set_ylabel("Амплитуда, LSB")
axs[0, 0].legend()

# --- График 2: тракт B ---
axs[0, 1].plot(t_plot, in_plot, color="gray", alpha=0.55, label="Вход 2000 Гц")
axs[0, 1].plot(t_B[:N_SHOW] * 1000, track_B_signed[:N_SHOW], "o-",
               color="blue", markersize=5, label="Тракт B (250 Гц)")
axs[0, 1].set_title("Тракт B — усреднение")
axs[0, 1].set_xlabel("Время, мс")
axs[0, 1].legend()

# --- График 3: спектр A ---
axs[1, 0].plot(freq_A, fft_A, color="red", label="Спектр A")
axs[1, 0].axvline(F_SIGNAL, color="white", linestyle=":", alpha=0.4)
axs[1, 0].set_title("Спектр тракта A")
axs[1, 0].set_xlabel("Частота, Гц")
axs[1, 0].set_ylabel("Амплитуда, дБ")
axs[1, 0].legend()

# --- График 4: спектр B + АЧХ boxcar ---
axs[1, 1].plot(freq_B, fft_B, color="blue", label="Спектр B")

axs[1, 1].axvline(F_SIGNAL, color="white", linestyle=":", alpha=0.4,
                  label=f"{F_SIGNAL:.0f} Гц")
axs[1, 1].set_title("Спектр тракта B")
axs[1, 1].set_xlabel("Частота, Гц")
axs[1, 1].legend()

for ax in axs.flat:
    ax.grid(True, alpha=0.22)

for ax in axs[1, :]:
    ax.set_xlim(0, FS_OUT / 2)
    ax.set_ylim(-60, 40)

plt.tight_layout()
plt.show()