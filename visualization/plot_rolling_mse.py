"""
Vẽ hình hai panel cho Chapter 2 (Phát biểu bài toán):
  - Panel trên : chuỗi tỷ giá OT + trung bình & độ lệch chuẩn trượt → drift rõ
  - Panel dưới : rolling MSE của mô hình đóng băng (noft) trong giai đoạn test
                 → sai số đạt đỉnh khi drift mạnh nhất (khủng hoảng 2008)

Checkpoint: Exchange_512_96 fewshot (fine-tuned on train, no online update)
Hình lưu: images_thesis/exchange-rolling-mse.pdf
Copy vào: images/exchange-drift.pdf trong report-v2 để LaTeX dùng.

Chạy từ OnlineTSF-TTM/:
    conda run -n proceed-ttm python images_thesis/plot_rolling_mse.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates

# ---------- Project imports ----------
from exp.exp_main import Exp_Main

# ---------- Args (from probe log Namespace, fewshot Exchange 512/96) ----------
ARGS = SimpleNamespace(
    train_only=False, wo_test=False, wo_valid=False, only_test=False, do_valid=False,
    model='TinyTimeMixer', override_hyper=True, compile=False, reduce_bs=False,
    normalization=None, checkpoints='./checkpoints/', tag='',
    pretrained_model_name='ibm-research/ttm-research-r2',
    backbone_mode='common_channel', decoder_mode='common_channel',
    run_offline=False, online_method=None, skip=None, online_learning_rate=None,
    val_online_lr=True, diff_online_lr=False, save_opt=True, leakage=False,
    debug=False, pretrain=False, freeze=False, freeze_online=False,
    act='sigmoid', tune_mode='down_up', ema=0, concept_dim=200, bottleneck_dim=32,
    individual_generator=False, share_encoder=False, use_mean=True,
    joint_update_valid=False, comment='', wo_clip=False,
    learning_rate_w=0.001, learning_rate_bias=0.001,
    border_type='online', train_ratio=0.5, test_ratio=0.45,
    root_path='./dataset/', dataset='Exchange', features='M', target='OT', freq='d',
    wrap_data_class=[], pin_gpu=True,
    seq_len=512, label_len=48, pred_len=96,
    individual=False, fc_dropout=0.05, head_dropout=0.0,
    patch_len=16, stride=8, padding_patch='end',
    revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25,
    drop_last=False, embed_type=0,
    d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
    moving_avg=25, factor=3, distil=True, dropout=0.05,
    embed='timeF', activation='gelu', output_attention=False, output_enc=False,
    do_predict=False, seg_len=24, win_size=2, num_routers=10,
    class_strategy='projection', subgraph_size=20, in_dim=1, gpt_layers=6,
    tmax=10, patch_size=16, num_workers=0, itr=1, train_epochs=100,
    begin_valid_epoch=0, batch_size=64, patience=3, optim='Adam',
    learning_rate=0.0001, des='test', loss='mse', lradj='type3',
    use_amp=False, pct_start=0.3, warmup_epochs=5,
    use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3',
    test_flop=False, local_rank=-1, test_train_num=500, selected_data_num=5,
    lambda_period=0.1, whole_model=False, continual=False, ensemble=False,
    revision='main', enc_in=8, c_out=8, data_path='exchange_rate.csv',
    dec_in=8, data='custom', ratio=(0.5, 0.45),
    model_id='Exchange_512_96_TinyTimeMixer', timeenc=2,
    find_unused_parameters=False,
)

CKPT = (
    './checkpoints/'
    'Exchange_512_96_TinyTimeMixer_online_ftM_sl512_ll48_pl96_lr0.0001'
    '_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'
)


# ── 1. Collect per-sample MSE (noft: frozen checkpoint, no online update) ─────

print('=== noft inference ===')
exp = Exp_Main(ARGS)

# Get borders from train data
train_data, _ = exp._get_data('train')
ARGS.borders = train_data.borders
exp.wrap_data_kwargs['borders'] = ARGS.borders

# Load checkpoint
exp.load_checkpoint(CKPT)
exp.model.eval()

test_data, test_loader = exp._get_data(flag='test')
val_end = ARGS.borders[0][2]   # index in raw data where val ends / test starts

mse_per_sample = []
pred_first_step = []    # pred[:, 0, :] per sample → first horizon step, normalized
with torch.no_grad():
    for batch in test_loader:
        outputs = exp.forward(batch)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        y = batch[1]
        for s in range(outputs.shape[0]):
            mse = F.mse_loss(outputs[s], y[s].to(exp.device)).item()
            mse_per_sample.append(mse)
            # First predicted step (t+1) in normalized space
            pred_first_step.append(outputs[s, 0, :].cpu().numpy())   # (C,)

noft_mse = np.array(mse_per_sample)   # (n_test,)
n_test = len(noft_mse)
print(f'  samples: {n_test}, mean MSE: {noft_mse.mean():.4f}')

# Inverse-transform predictions → original scale (scaler fitted on train only)
pred_first_step = np.array(pred_first_step)          # (n_test, C)
pred_inv = test_data.inverse_transform(pred_first_step)  # (n_test, C)

# OT is the last column in the dataset (index -1)
OT_COL = -1
pred_ot = pred_inv[:, OT_COL]   # (n_test,)  — model's 1-step-ahead forecast for OT

# Save raw arrays
np.save(os.path.join(os.path.dirname(__file__), 'noft_mse_exchange_512_96.npy'), noft_mse)
np.save(os.path.join(os.path.dirname(__file__), 'noft_pred_ot_exchange_512_96.npy'), pred_ot)


# ── 2. Map sample indices → forecast-start dates ──────────────────────────────

df_raw = pd.read_csv('./dataset/exchange_rate.csv', parse_dates=['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)
dates_all = df_raw['date'].values   # numpy array of datetime64

# Each test sample i predicts from position: val_end + i + seq_len
forecast_dates = dates_all[val_end + ARGS.seq_len : val_end + ARGS.seq_len + n_test]
forecast_dates = pd.to_datetime(forecast_dates)

# Full series dates & values for panel 1
y_full  = df_raw['OT'].values
d_full  = pd.to_datetime(dates_all)


# ── 3. Rolling statistics ──────────────────────────────────────────────────────

STAT_WIN = 365    # ~1 year

y_ser = pd.Series(y_full, index=d_full)
roll_mean = y_ser.rolling(f'{STAT_WIN}D', min_periods=90, center=True).mean()
roll_std  = y_ser.rolling(f'{STAT_WIN}D', min_periods=90, center=True).std()

MSE_WIN = 180     # ~6 months rolling for error panel
mse_ser = pd.Series(noft_mse, index=forecast_dates)
roll_mse = mse_ser.rolling(f'{MSE_WIN}D', min_periods=30, center=True).mean()


# ── 4. Figure ──────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5, 'figure.dpi': 150,
})

fig, axes = plt.subplots(2, 1, figsize=(11, 5.8),
                         gridspec_kw={'height_ratios': [2, 1.3], 'hspace': 0.05},
                         sharex=True)
ax1, ax2 = axes

# High-drift period: identified from roll_mse peak region (actual data)
HIGH_DRIFT_START = pd.Timestamp('2001-09-01')   # AUD at historic lows
HIGH_DRIFT_END   = pd.Timestamp('2003-09-01')   # rapid AUD appreciation

# X limits: test period only (aligned with panel 2)
X_MIN = forecast_dates.min()
X_MAX = forecast_dates.max()

# Panel 1: zoomed into test period — same x-window as panel 2
ax1.plot(d_full, y_full, color='#4C72B0', lw=0.9, alpha=0.6, zorder=1,
         label='Tỷ giá OT thực tế')
ax1.plot(roll_mean.index, roll_mean.values, color='#DD8452', lw=1.8, zorder=3,
         label=f'Trung bình trượt {STAT_WIN} ngày')
ax1.fill_between(roll_mean.index,
                 (roll_mean - roll_std).values, (roll_mean + roll_std).values,
                 alpha=0.18, color='#DD8452', zorder=2, label='±1 std trượt')

# Dự báo của mô hình đóng băng (bước đầu tiên, inverse-transform về thang gốc)
ax1.plot(forecast_dates, pred_ot, color='#CC3311', lw=1.1, ls='--',
         alpha=0.85, zorder=4, label='Dự báo mô hình đóng băng (bước t+1)')

ax1.axvspan(HIGH_DRIFT_START, HIGH_DRIFT_END, alpha=0.15, color='#8B0000', zorder=0)

ax1.set_ylabel('Giá trị tỷ giá OT')
ax1.legend(loc='upper left', framealpha=0.88, ncol=2, fontsize=7.5)
ax1.set_title(
    'Biến đổi đặc tính thống kê trong dữ liệu tỷ giá hối đoái (Exchange) — giai đoạn kiểm tra',
    fontsize=10.5, pad=5)

# Panel 2: noft rolling MSE over test period
ax2.plot(mse_ser.index, mse_ser.values,
         color='#D62728', alpha=0.10, lw=0.4)
ax2.plot(roll_mse.index, roll_mse.values,
         color='#D62728', lw=1.8,
         label=f'MSE trung bình trượt {MSE_WIN} ngày — mô hình đóng băng')
ax2.axvspan(HIGH_DRIFT_START, HIGH_DRIFT_END, alpha=0.18, color='#8B0000', zorder=0,
            label='Giai đoạn drift cao (tỷ giá biến đổi mạnh)')

# Annotate peak
peak_idx = roll_mse.idxmax()
peak_val = roll_mse.max()
ax2.annotate(f'Đỉnh sai số\n{peak_idx.strftime("%m/%Y")}\nMSE={peak_val:.2f}',
             xy=(peak_idx, peak_val),
             xytext=(peak_idx + pd.DateOffset(years=2), peak_val * 0.85),
             arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.0),
             fontsize=7.5, color='#8B0000', va='center')

ax2.set_ylabel('MSE trung bình trượt')
ax2.set_xlabel('Thời gian')
ax2.legend(loc='upper right', framealpha=0.88)
ax2.set_ylim(bottom=0)

# Format x-axes — sharex=True means setting xlim on ax2 propagates to ax1
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', ls=':', lw=0.4, alpha=0.5)

ax2.set_xlim(X_MIN, X_MAX)
ax1.set_xticklabels([])

plt.tight_layout()
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exchange-rolling-mse')
fig.savefig(base + '.pdf', dpi=180, bbox_inches='tight')
fig.savefig(base + '.png', dpi=180, bbox_inches='tight')
print(f'\nSaved: {base}.pdf')
print(f'Saved: {base}.png')

# Summary stats for LaTeX caption
valid = ~np.isnan(roll_mse.values)
rvals = roll_mse.values[valid]
print(f'MSE range: {rvals.min():.4f} – {rvals.max():.4f}')
print(f'Peak date: {roll_mse.idxmax().strftime("%Y-%m")}')
print(f'Mean noft MSE: {noft_mse.mean():.4f}')
