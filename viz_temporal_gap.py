"""
Minh hoa "khoang trong thoi gian" (temporal gap) tren du lieu that (Exchange / OT).
Bo cuc 3 panel tach biet de KHONG bi chong chu:
  Panel 1: chuoi that + trung binh truot + nen che do + ngoac khoang trong H
  Panel 2: so do cac cua so (recent / unusable / current) + co che delta -> Delta theta
  Panel 3: trung binh & do lech chuan truot (drift = dai luong do duoc)

Chay:  /home/daoviethoang/.conda/envs/proceed-ttm/bin/python viz_temporal_gap.py
Xuat:  images_thesis/temporal_gap.png  (+ .pdf vector)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ----------------------------------------------------------------------------
# 1. Du lieu that
# ----------------------------------------------------------------------------
df = pd.read_csv("dataset/exchange_rate.csv")
series = df["OT"].values.astype(float)
dates = pd.to_datetime(df["date"], format="mixed")

L, H = 200, 140                                   # do dai minh hoa (nho cho de nhin)
t = 4894                                          # cuoi cua so dau vao hien tai (chi so global)
disp_lo = t - L - H - 130
right_pad = 300                                    # cot trong ben phai cho hop co che
disp_hi = t + H + right_pad
xs = np.arange(disp_lo, disp_hi)
ys = series[disp_lo:disp_hi]

win = 90
roll_mean = pd.Series(series).rolling(win, center=True).mean().values[disp_lo:disp_hi]
roll_std = pd.Series(series).rolling(win, center=True).std().values[disp_lo:disp_hi]

# Moc cua so (theo cong thuc no-leakage)
cur_in, cur_out = (t - L, t), (t, t + H)
rec_in, rec_out = (t - H - L, t - H), (t - H, t)
gap = (t - H, t)
boundary = t - H

# ----------------------------------------------------------------------------
# 2. Mau & style
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8})
C_OLD, C_NEW = "#F2C14E", "#B39CD0"
C_SERIES, C_MEAN = "#37474F", "#111111"
C_REC, C_UNUSABLE, C_CUR, C_GAP = "#2E7D32", "#9E9E9E", "#1565C0", "#C62828"

fig, (ax, axm, ax2) = plt.subplots(
    3, 1, figsize=(13.5, 10.6), height_ratios=[2.5, 1.75, 1.0], sharex=True,
    gridspec_kw=dict(hspace=0.10))

# ============================ PANEL 1: chuoi that ============================
ax.axvspan(disp_lo, boundary, color=C_OLD, alpha=0.16, zorder=0)
ax.axvspan(boundary, disp_hi, color=C_NEW, alpha=0.16, zorder=0)
ax.plot(xs, ys, color=C_SERIES, lw=0.9, alpha=0.85, zorder=2,
        label="Chuỗi quan sát (Exchange, biến OT)")
ax.plot(xs, roll_mean, color=C_MEAN, lw=2.4, zorder=3, label="Trung bình trượt")

ymin, ymax = np.nanmin(ys), np.nanmax(ys)
span = ymax - ymin

# nhan che do - dat sat day panel, khong dung vung tren (danh cho ngoac)
ax.text((disp_lo + boundary) / 2, ymin + 0.06 * span,
        "Chế độ dữ liệu A\n(đặc tính thống kê cũ)", ha="center", va="bottom",
        fontsize=11, color="#7A5B00", weight="bold")
ax.text((boundary + disp_hi) / 2 + 20, ymin + 0.06 * span,
        "Chế độ dữ liệu B\n(đặc tính đã dịch chuyển)", ha="center", va="bottom",
        fontsize=11, color="#4A356B", weight="bold")

# ngoac khoang trong - o dinh panel, co headroom rieng
y_gap = ymax + 0.13 * span
ax.annotate("", xy=(gap[0], y_gap), xytext=(gap[1], y_gap),
            arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=2.2), clip_on=False)
for xb in gap:
    ax.plot([xb, xb], [ymax + 0.02 * span, y_gap], color=C_GAP, lw=1.0, ls=":", alpha=0.7)
ax.text((gap[0] + gap[1]) / 2, y_gap + 0.03 * span,
        r"KHOẢNG TRỐNG THỜI GIAN  $H$" "\n"
        "biến đổi xảy ra ở đây — nhưng tham số KHÔNG được cập nhật",
        ha="center", va="bottom", fontsize=10.5, color=C_GAP, weight="bold")

ax.set_ylim(ymin - 0.05 * span, ymax + 0.30 * span)
ax.set_ylabel("Giá trị\n(đã chuẩn hóa gốc)", fontsize=10.5)
ax.set_yticks([])
ax.legend(loc="center left", fontsize=9.5, framealpha=0.9,
          bbox_to_anchor=(0.005, 0.42))

# ==================== PANEL 2: so do cua so + co che ========================
axm.axvspan(disp_lo, boundary, color=C_OLD, alpha=0.10, zorder=0)
axm.axvspan(boundary, disp_hi, color=C_NEW, alpha=0.10, zorder=0)
axm.set_ylim(0, 1)

BAR_H = 0.11
def bar(x0x1, y, color, alpha=1.0, ec=None, lw=1.0, ls="-"):
    a, b = x0x1
    axm.add_patch(FancyBboxPatch((a, y), b - a, BAR_H, boxstyle="square,pad=0",
                  facecolor=color, alpha=alpha, edgecolor=ec or color, lw=lw, ls=ls, zorder=4))

# --- hang 1: cap huan luyen gan nhat (chu thich NGAY DUOI thanh) ---
yr = 0.82
bar(rec_in, yr, C_REC, 0.9)
bar(rec_out, yr, C_REC, 0.45, ec=C_REC, lw=1.3)
axm.text(rec_in[0] - 12, yr + BAR_H / 2, r"$X_{t-H}$", ha="right", va="center",
         fontsize=11, color=C_REC, weight="bold")
axm.text(rec_out[1] + 12, yr + BAR_H / 2, r"$Y_{t-H}$", ha="left", va="center",
         fontsize=11, color=C_REC, weight="bold")
axm.text((rec_in[0] + rec_out[1]) / 2, yr - 0.055,
         "Cặp huấn luyện gần nhất — đã có ĐỦ nhãn", ha="center", va="top",
         fontsize=9.5, color=C_REC, weight="bold")

# --- hang 2: cac cua so da quan sat nhung chua co nhan ---
yu = 0.50
n_un = 5
for k in range(n_un):
    x0 = (t - H) + 12 + k * 24
    bar((x0, x0 + L * 0.9), yu + (k - n_un / 2) * 0.012, C_UNUSABLE, 0.34 - 0.045 * k)
axm.text(t - H - 12, yu + BAR_H / 2, r"$X_{t-H+1},\ldots$", ha="right", va="center",
         fontsize=10, color="#616161", weight="bold")
axm.text((t - H + t) / 2 + 30, yu - 0.055,
         "Đã quan sát nhưng CHƯA có nhãn  ⇒  không dùng cập nhật được",
         ha="center", va="top", fontsize=9.5, color="#616161", style="italic")

# --- hang 3: cua so hien tai ---
yc = 0.18
bar(cur_in, yc, C_CUR, 0.92)
bar(cur_out, yc, "none", ec=C_CUR, lw=1.7, ls=(0, (4, 3)))
axm.text(cur_in[0] - 12, yc + BAR_H / 2, r"$X_{t}$", ha="right", va="center",
         fontsize=11.5, color=C_CUR, weight="bold")
axm.text(cur_out[1] + 12, yc + BAR_H / 2, r"$\hat{Y}_{t}$", ha="left", va="center",
         fontsize=11.5, color=C_CUR, weight="bold")
axm.text((cur_in[0] + cur_out[1]) / 2, yc - 0.055,
         "Cửa sổ hiện tại: đầu vào đã có, cần dự báo (nhãn chưa có)",
         ha="center", va="top", fontsize=9.5, color=C_CUR, weight="bold")

# --- moc truc & nhan t-H, t, t+H ---
for xv, lab in [(t - H, r"$t-H$"), (t, r"$t$"), (t + H, r"$t+H$")]:
    axm.axvline(xv, color="#777777", lw=0.8, ls=":", zorder=1)
    axm.text(xv, 0.985, lab, ha="center", va="top", fontsize=11,
             color="#222222", weight="bold")

axm.set_ylabel("Sơ đồ\ncác cửa sổ", fontsize=10.5)
axm.set_yticks([])
for s in ("top", "right", "left"):
    axm.spines[s].set_visible(False)

# --- co che: -> delta -> Delta theta (cot trong ben phai t+H, khong dinh thanh) ---
mx = t + H + 60
bw = right_pad - 110
box_d = FancyBboxPatch((mx, 0.58), bw, 0.30, boxstyle="round,pad=0.02",
                       facecolor="#FFF3E0", edgecolor="#E65100", lw=1.5, zorder=6)
box_t = FancyBboxPatch((mx, 0.14), bw, 0.30, boxstyle="round,pad=0.02",
                       facecolor="#E3F2FD", edgecolor="#0D47A1", lw=1.5, zorder=6)
axm.add_patch(box_d); axm.add_patch(box_t)
axm.text(mx + bw / 2, 0.73, r"$\delta_{t-H\to t}$" "\nđộ dịch chuyển",
         ha="center", va="center", fontsize=10, color="#E65100", weight="bold", zorder=7)
axm.text(mx + bw / 2, 0.29, r"$\Delta\theta$" "\nhiệu chỉnh\ntham số",
         ha="center", va="center", fontsize=10, color="#0D47A1", weight="bold", zorder=7)
axm.add_patch(FancyArrowPatch((mx + bw / 2, 0.58), (mx + bw / 2, 0.44),
              arrowstyle="-|>", mutation_scale=15, color="#333333", lw=1.5, zorder=7))
# mui ten tu X_t va cap gan nhat vao hop delta (ngan, khong cat ngang)
axm.annotate("", xy=(mx - 6, 0.70), xytext=(rec_out[1] + 45, yr),
             arrowprops=dict(arrowstyle="-|>", color="#E9964E", lw=1.2,
                             connectionstyle="arc3,rad=-0.15"), zorder=5)
axm.annotate("", xy=(mx - 6, 0.64), xytext=(cur_out[0] + 30, yc + BAR_H),
             arrowprops=dict(arrowstyle="-|>", color="#E9964E", lw=1.2,
                             connectionstyle="arc3,rad=0.20"), zorder=5)

axm.set_xlim(disp_lo, disp_hi)

# ==================== PANEL 3: drift do duoc ========================
ax2.plot(xs, roll_mean, color=C_MEAN, lw=1.8, label="Trung bình trượt (90 ngày)")
ax2.fill_between(xs, roll_mean - roll_std, roll_mean + roll_std,
                 color=C_SERIES, alpha=0.15, label=r"$\pm$ độ lệch chuẩn trượt")
ax2.axvspan(disp_lo, boundary, color=C_OLD, alpha=0.10, zorder=0)
ax2.axvspan(boundary, disp_hi, color=C_NEW, alpha=0.10, zorder=0)
for xv in (t - H, t, t + H):
    ax2.axvline(xv, color="#777777", lw=0.8, ls=":")
ax2.set_ylabel("Thống kê\ncửa sổ", fontsize=10.5)
ax2.legend(loc="upper right", fontsize=8.8, framealpha=0.9)

tick_idx = np.linspace(disp_lo, disp_hi - 1, 6).astype(int)
ax2.set_xticks(tick_idx)
ax2.set_xticklabels([dates.iloc[i].strftime("%m/%Y") for i in tick_idx], fontsize=9)
ax2.set_xlabel("Thời gian  →  (trung bình & phương sai dịch chuyển rõ qua khoảng trống "
               "⇒ 'drift' là đại lượng đo được)", fontsize=10)

fig.suptitle("Khoảng trống thời gian trong dự báo chuỗi thời gian trực tuyến\n"
             "(dữ liệu thật: Exchange – biến OT, ~2002–2004)",
             fontsize=13.5, weight="bold", y=0.985)
fig.text(0.5, 0.003,
         f"Ghi chú: độ dài cửa sổ minh hoạ L={L}, H={H} (chọn nhỏ cho dễ nhìn); "
         f"thí nghiệm thực dùng L∈{{512,1024,1536}}, H∈{{96,192,336,720}}.",
         ha="center", fontsize=8.5, color="#666666")

os.makedirs("images_thesis", exist_ok=True)
fig.savefig("images_thesis/temporal_gap.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("images_thesis/temporal_gap.pdf", bbox_inches="tight", facecolor="white")
print("saved images_thesis/temporal_gap.png (+.pdf)")
