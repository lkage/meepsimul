#!/usr/bin/env bash
# 논문 Fig. 2(b) (로드가 "1:2" 위치에 있는 경우)와 Fig. 3(b) (로드가 정중앙
# "1:1" 위치에 있는 경우)를 재현하는 데 필요한 시뮬레이션들을 전부 실행한 뒤,
# 투과율 스펙트럼을 겹쳐서 그래프로 그려주는 스크립트.
#
# 주의: 여기 기본값(RESOLUTION=60)은 "스크립트가 잘 도는지" 빠르게 확인하는
# 저해상도 스모크테스트용이지, 논문 수준의 정밀한 재현용이 아니다 (자세한
# 내용은 README.md 참고). 실제로 신뢰할 만한 결과를 얻으려면 RESOLUTION을
# 250 이상으로 올리고 mpirun의 랭크(프로세스) 수도 훨씬 늘려야 한다. 3D
# 시뮬레이션이라 해상도를 올릴수록 계산 시간과 메모리 사용량이 매우 가파르게
# 증가하니 참고할 것.
set -euo pipefail

NP="${NP:-4}"                 # MPI 병렬 프로세스(랭크) 개수
RESOLUTION="${RESOLUTION:-60}"  # 1 um 당 격자점 수
NFREQ="${NFREQ:-400}"           # 스펙트럼에서 기록할 주파수 점 개수
COMMON="--resolution $RESOLUTION --nfreq $NFREQ"

# run(): simulate.py를 mpirun으로 실행하기 전에, 실행할 명령을 화면에 먼저
# 출력해 주는 작은 헬퍼 함수. "$@"는 run 호출 시 넘긴 인자들을 그대로
# simulate.py에 전달한다는 뜻.
run() {
  echo "+++ $*"
  mpirun -np "$NP" python3 simulate.py "$@"
}

# 두 가지 로드 위치(1:2 / 1:1)에 대해 각각 필요한 4개의 실행을 반복한다.
for POS_DIR in pos12 pos11; do
  if [ "$POS_DIR" = pos12 ]; then POSITION=1:2; else POSITION=1:1; fi
  OUTDIR="results/$POS_DIR"
  mkdir -p "$OUTDIR"

  # 1) 입사 플럭스 레퍼런스 (구조물을 전혀 넣지 않은 진공 전파).
  #    이후 투과율 계산 시 나눗셈의 분모가 되는 기준값.
  run --mode spectrum --empty $COMMON --outdir "$OUTDIR"

  # 2) 로드 없이 공진기(슬롯)만 있는 경우 -- "맨 슬롯" 기준 스펙트럼
  run --mode spectrum --rod none $COMMON --outdir "$OUTDIR"

  # 3) 이 위치에 폭 600 nm짜리("large") Pt 로드를 놓은 경우
  run --mode spectrum --rod large --position "$POSITION" $COMMON --outdir "$OUTDIR"

  # 4) 이 위치에 폭 250 nm짜리("small") Pt 로드를 놓은 경우
  run --mode spectrum --rod small --position "$POSITION" $COMMON --outdir "$OUTDIR"

  # 위 4개 실행 결과를 겹쳐 그린 그래프 + 콘솔에 후보 공진 주파수 출력
  python3 analyze.py spectrum --indir "$OUTDIR" --outfile "$OUTDIR/spectrum.png"
done

echo
echo "Spectra written to results/pos12/spectrum.png and results/pos11/spectrum.png."
echo "To reproduce the |Ex|^2 field maps (Fig. 2(c)/3(c)), read off the printed"
echo "1st/3rd-harmonic candidate frequencies above and run, e.g.:"
echo
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod none \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod large --position 1:2 \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod small --position 1:2 \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  python3 analyze.py field --indir results/pos12 --pattern 'field_*.npy' \\"
echo "      --outfile results/pos12/fields.png"
