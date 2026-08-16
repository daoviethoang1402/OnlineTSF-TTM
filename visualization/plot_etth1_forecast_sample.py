"""
Vẽ một mẫu dự báo thực tế của mô hình cơ sở (TTM-only-noft, không có PROCEED)
trên bộ dữ liệu ETTh1, dữ liệu đầu vào 512 giờ, dữ liệu cần dự đoán 96 giờ.

Với mỗi biến trong 7 biến (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT):
  - đường liền màu đỏ: toàn bộ giá trị thực tế, từ đầu dữ liệu đầu vào đến hết dữ liệu cần dự đoán
  - đường đứt nét màu xanh: dự báo của mô hình cho 96 bước tương lai
  - đường thẳng đứng: mốc "thời điểm hiện tại", ranh giới giữa dữ liệu đầu vào (input)
    và dữ liệu cần dự đoán (output)

Checkpoint: ETTh1_512_96 TTM-only-noft (huấn luyện lại với đúng seed/hyperparameter
trong logs/probe_forget/fewshot_ETTh1_512_96.log, vì checkpoint gốc đã bị dọn dẹp).
Hình lưu: images_thesis/etth1-forecast-sample.png

Chạy từ OnlineTSF-TTM/:
    conda run -n proceed-ttm python images_thesis/plot_etth1_forecast_sample.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp.exp_main import Exp_Main

# ---------- Args (khớp Namespace trong logs/probe_forget/fewshot_ETTh1_512_96.log) ----------
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
    root_path='./dataset/', dataset='ETTh1', features='M', target='OT', freq='h',
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
    revision='main', enc_in=7, c_out=7, data_path='ETTh1.csv', dec_in=7, data='ETTh1',
    ratio=(0.5, 0.45), model_id='ETTh1_512_96_TinyTimeMixer', timeenc=2,
    find_unused_parameters=False,
)

CKPT = (
    './checkpoints/'
    'ETTh1_512_96_TinyTimeMixer_online_ftM_sl512_ll48_pl96_lr0.0001'
    '_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'
)

CHANNEL_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# ── 1. Nạp mô hình TTM-only-noft đã huấn luyện ─────────────────────────────

print('=== Nạp checkpoint TTM-only-noft (ETTh1, 512/96) ===')
exp = Exp_Main(ARGS)

train_data, _ = exp._get_data('train')
ARGS.borders = train_data.borders
exp.wrap_data_kwargs['borders'] = ARGS.borders

exp.load_checkpoint(CKPT)
exp.model.eval()

test_data, _ = exp._get_data(flag='test')
n_test = len(test_data)
print(f'  Số mẫu test: {n_test}')

# ── 2. Chọn một mẫu đại diện (giữa tập test) và chạy suy luận ──────────────

sample_idx = n_test // 2
x, y, x_mark, y_mark = test_data[sample_idx]  # x: (L,C) ; y: (H,C)

batch = [x.unsqueeze(0), y.unsqueeze(0), x_mark.unsqueeze(0), y_mark.unsqueeze(0)]
with torch.no_grad():
    outputs = exp.forward(batch)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

pred_norm = outputs[0].cpu().numpy()          # (H, C), thang chuẩn hóa
x_np = x.cpu().numpy()                        # (L, C), thang chuẩn hóa
y_np = y.cpu().numpy()                        # (H, C), thang chuẩn hóa

L, H, C = ARGS.seq_len, ARGS.pred_len, ARGS.enc_in

# Đưa về thang gốc (đơn vị thật) bằng scaler đã fit trên tập huấn luyện
truth_norm = np.concatenate([x_np, y_np], axis=0)          # (L+H, C)
truth = test_data.inverse_transform(truth_norm)             # (L+H, C)
pred = test_data.inverse_transform(pred_norm)                # (H, C)

sample_mse = float(np.mean((pred_norm - y_np) ** 2))
print(f'  Mẫu test index={sample_idx}, MSE (thang chuẩn hóa) = {sample_mse:.4f}')

# ── 3. Vẽ hình: 7 subplot, mỗi subplot một biến ────────────────────────────

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
})

t_full = np.arange(L + H)
t_pred = np.arange(L, L + H)
now_boundary = L - 0.5

fig, axes = plt.subplots(C, 1, figsize=(10, 2.1 * C), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(t_full, truth[:, i], color='#D62728', lw=1.1, alpha=0.9,
            label='Giá trị thực tế (dữ liệu đầu vào + dữ liệu cần dự đoán)')
    ax.plot(t_pred, pred[:, i], color='#1F77B4', lw=1.4, ls='--',
            label='Dự báo của mô hình cơ sở (TTM-only-noft)')
    ax.axvline(now_boundary, color='black', lw=1.0, ls='-', alpha=0.7,
               label='Thời điểm hiện tại (ranh giới dữ liệu đầu vào / dữ liệu cần dự đoán)')
    ax.set_title(CHANNEL_NAMES[i], fontsize=9.5, fontweight='bold', loc='left', pad=3)
    ax.set_ylabel('Giá trị')
    ax.grid(which='major', ls=':', lw=0.4, alpha=0.5)

axes[0].legend(loc='upper left', framealpha=0.9, fontsize=6.6, ncol=1)
axes[-1].set_xlabel('Thời gian (giờ, chỉ số bước trong mẫu)')
axes[-1].set_xlim(0, L + H - 1)

fig.suptitle(
    'Một mẫu dự báo của mô hình cơ sở TTM-only-noft trên ETTh1\n'
    '(dữ liệu đầu vào 512 giờ, dữ liệu cần dự đoán 96 giờ, 7 biến)',
    fontsize=11, y=0.998,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etth1-forecast-sample.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f'\nSaved: {out}')
