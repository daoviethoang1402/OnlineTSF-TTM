#!/usr/bin/env bash
# =============================================================================
# Kiem tra & don dep truoc khi RESUME mot session bi ngat giua chung.
#
# VAN DE: cac script driver ghi log bang '>>' (append). Logic skip chi bo qua khi
# log da co du ITR dong 'mse:'. Nen:
#   * log co >= ITR dong 'mse:'  -> DONE, se duoc skip                    [OK]
#   * log co 0 dong 'mse:'       -> se chay lai, append vao rac cu        [ban, nhung khong sai]
#   * log co 1..ITR-1 dong       -> se chay lai va APPEND them ITR dong nua
#                                   => tong 4-5 dong 'mse:' cho 3 seed
#                                   => HONG DU LIEU khi parse             [NGUY HIEM]
# Script nay liet ke trang thai va (voi --clean) XOA cac log chua hoan tat,
# de lan chay lai bat dau tu file sach.
#
# Dung:
#   bash ttm-scripts/ablation-study-2/resume_check.sh           # chi bao cao
#   bash ttm-scripts/ablation-study-2/resume_check.sh --clean   # bao cao + xoa log dang do
# =============================================================================
set -u
ITR=${ITR:-3}
LOG_DIR=${LOG_DIR:-logs/ablation-study-2}
PATTERN=${PATTERN:-"ssffinal_*_itr${ITR}.log"}
CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1

shopt -s nullglob
files=("$LOG_DIR"/$PATTERN)
if [ ${#files[@]} -eq 0 ]; then
  echo "Khong tim thay log nao khop: $LOG_DIR/$PATTERN"
  echo "  -> Neu chay tren Kaggle: kiem tra xem /kaggle/working con du lieu khong."
  exit 0
fi

done_n=0; partial_n=0; empty_n=0
partial_files=(); empty_files=()

for f in "${files[@]}"; do
  n=$(grep -c 'mse:' "$f" 2>/dev/null || true)
  n=${n:-0}
  if [ "$n" -ge "$ITR" ]; then
    done_n=$((done_n+1))
  elif [ "$n" -gt 0 ]; then
    partial_n=$((partial_n+1)); partial_files+=("$f")
    echo "  DANG DO ($n/$ITR seed): $f"
  else
    empty_n=$((empty_n+1)); empty_files+=("$f")
    echo "  CHUA CO KET QUA (0/$ITR): $f"
  fi
done

echo
echo "Tong ket: $done_n hoan tat / $partial_n dang do / $empty_n chua co ket qua"
echo "          (tong $((done_n+partial_n+empty_n)) log)"

if [ $((partial_n+empty_n)) -eq 0 ]; then
  echo "Tat ca da hoan tat -- khong can lam gi."
  exit 0
fi

if [ "$CLEAN" -eq 1 ]; then
  for f in "${partial_files[@]}" "${empty_files[@]}"; do rm -f "$f"; done
  echo "Da xoa $((partial_n+empty_n)) log chua hoan tat."
  echo "Gio chay lai dung 4 script cu -- chung se skip $done_n cau hinh da xong."
else
  echo
  echo "CAN XOA $partial_n log dang do truoc khi chay lai (neu khong se bi append"
  echo "them $ITR dong 'mse:' nua -> sai so seed khi parse). Chay:"
  echo "    bash ttm-scripts/ablation-study-2/resume_check.sh --clean"
fi
