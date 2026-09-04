#!/usr/bin/env python3
"""simulate.py가 만든 결과를 후처리/시각화하는 스크립트.

두 개의 서브커맨드를 제공한다:

  spectrum  simulate.py --mode spectrum 이 만든 flux_*.npz 파일들을
            디렉터리에서 불러와, --empty 레퍼런스 실행 결과로 나눠서
            투과율(transmittance)을 계산하고, "로드 없음 / 큰 로드 /
            작은 로드" 세 곡선을 한 그래프에 겹쳐 그린다 -- 논문
            Fig. 2(b) / Fig. 3(b) 스타일의 재현. 더불어 각 곡선에서
            가장 두드러진 피크 주파수(1차/3차 하모닉 후보)를 찾아
            콘솔에 출력한다.

  field     simulate.py --mode field 가 만든 field_*.npy 파일들
            (|Ex|^2 공간 분포)을 불러와 가짜색상(false-color) 이미지로
            그린다 -- 논문 Fig. 2(c) / Fig. 3(c) 스타일의 재현.

사용 예:
    python3 analyze.py spectrum --indir results/pos12 --outfile fig2b.png
    python3 analyze.py field --indir results/pos12 --pattern 'field_*_f0.05*.npy' \
        --outfile fig2c_fundamental.png
"""

import argparse
import glob
import os

import numpy as np
import matplotlib

# "Agg" 백엔드: 화면(디스플레이) 없이도 그래프를 파일로 저장할 수 있게 해주는
# matplotlib의 비대화형(non-interactive) 렌더링 백엔드. 서버/터미널 환경에서
# 그래프를 파일로만 저장할 때 흔히 쓰는 설정이다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cmd_spectrum(args):
    """투과율 스펙트럼 겹쳐그리기(overlay plot)를 만드는 서브커맨드."""

    # 1) 레퍼런스(입사 플럭스) 파일을 먼저 읽는다. 이게 없으면 투과율을
    #    계산할 기준값이 없으므로 곧바로 에러를 낸다.
    ref_file = os.path.join(args.indir, "flux_reference.npz")
    if not os.path.exists(ref_file):
        raise SystemExit(
            f"missing {ref_file}. Run `simulate.py --mode spectrum --empty` first "
            f"to produce the incident-flux reference."
        )
    ref = np.load(ref_file, allow_pickle=True)
    ref_freqs, ref_flux = ref["freqs"], ref["flux"]

    # 2) simulate.py의 make_run_tag()가 만드는 태그 -> 그래프 범례 문구 매핑.
    #    "1:2" 위치와 "1:1" 위치용 태그를 모두 나열해 두고, 실제로 존재하는
    #    파일만 아래 루프에서 골라 그린다 (즉 이 딕셔너리에 있다고 해서
    #    전부 그려지는 게 아니라, 해당 파일이 있을 때만 그려진다).
    labels = {
        "norod": "without nano rod",
        "large_pos12": "600 nm-width nano rod",
        "small_pos12": "250 nm-width nano rod",
        "large_pos11": "600 nm-width nano rod",
        "small_pos11": "250 nm-width nano rod",
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    found_any = False
    for tag, label in labels.items():
        f = os.path.join(args.indir, f"flux_{tag}.npz")
        if not os.path.exists(f):
            continue  # 이 조합의 실행 결과가 아직 없으면 건너뜀
        found_any = True
        d = np.load(f, allow_pickle=True)
        freqs, flux = d["freqs"], d["flux"]
        # 레퍼런스 실행과 이 실행이 같은 주파수 격자에서 계산됐는지 확인.
        # (--fcen/--df/--nfreq를 실행마다 다르게 주면 주파수 점이 어긋나서
        # 나눗셈이 의미 없어지므로, 모든 실행에 동일한 값을 써야 한다.)
        if not np.allclose(freqs, ref_freqs):
            raise SystemExit(f"{f} has different frequency grid than the reference run")

        # --- 핵심: 투과율 = (구조가 있을 때 투과 플럭스) / (구조 없을 때 입사 플럭스) ---
        # 이것은 THz-TDS(시간영역분광법) 실험에서 "샘플 스캔 / 레퍼런스 스캔"으로
        # 투과율을 얻는 것과 동일한 개념이다.
        T = flux / ref_flux
        ax.plot(freqs, T, label=label)

        # 3) 대략적인 공진 피크 찾기 (scipy가 있을 때만; 없으면 조용히 건너뜀).
        #    find_peaks: 국소적으로 주변보다 값이 큰 지점(피크)들을 찾는 함수.
        #    height=T.max()*0.15: 전체 최댓값의 15% 이상인 피크만 유의미하다고
        #    취급해서, 잡음 수준의 자잘한 피크는 걸러낸다.
        try:
            from scipy.signal import find_peaks

            idx, _ = find_peaks(T, height=T.max() * 0.15)
            if len(idx):
                # 피크들을 투과율 크기 내림차순으로 정렬해서, "가장 강한 피크가
                # 먼저 나오도록" 출력한다 -- 보통 이게 기본(fundamental) 모드다.
                peak_f = freqs[idx][np.argsort(T[idx])[::-1]]
                print(f"[{tag}] candidate resonance frequencies (1/um), "
                      f"strongest first: {np.round(peak_f[:5], 4).tolist()}")
        except ImportError:
            pass

    if not found_any:
        raise SystemExit(f"no flux_*.npz runs found in {args.indir} (besides the reference)")

    ax.set_xlabel("Frequency (1/um, Meep units)")
    ax.set_ylabel("Transmittance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=150)
    print(f"[analyze.py] wrote {args.outfile}")


def cmd_field(args):
    """|Ex|^2 근접장 맵들을 나란히 놓고 그리는 서브커맨드."""

    # glob으로 --pattern에 매칭되는 파일들을 모두 찾는다 (예: 특정 주파수의
    # "로드없음/큰로드/작은로드" 세 파일을 한 번에 비교하고 싶을 때 사용).
    files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not files:
        raise SystemExit(f"no files matched {args.pattern!r} in {args.indir}")

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    # 여러 서브플롯이 같은 컬러 스케일(vmin~vmax)을 공유하도록, 전체 파일
    # 중 최댓값을 미리 구해 둔다. 그래야 "이 그림이 저 그림보다 세기가
    # 크다/작다"를 색만 보고도 비교할 수 있다 (논문 Fig. 2(c)처럼).
    vmax = max(np.load(f).max() for f in files)
    for ax, f in zip(axes[0], files):
        data = np.load(f)  # shape (nx, ny): 모니터 평면 위의 |Ex|^2 값
        # data.T로 전치하는 이유: imshow는 배열의 첫 축을 세로(행, y축 픽셀)로
        # 그리는데, 우리는 저장할 때 (x, y) 순서였으므로 전치해서 y가 세로로
        # 나오게 맞춘다 (아래 xlabel/ylabel 표기와 일치시키기 위함).
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(os.path.basename(f), fontsize=8)
        ax.set_xlabel("x index (width)")
        ax.set_ylabel("y index (length)")
    fig.colorbar(im, ax=axes[0].tolist(), label="|Ex|^2 (a.u.)", shrink=0.8)
    fig.savefig(args.outfile, dpi=150)
    print(f"[analyze.py] wrote {args.outfile}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 서브커맨드(spectrum / field) 구조: `python3 analyze.py spectrum ...`처럼
    # 첫 번째 위치 인자로 어떤 동작을 할지 고르게 한다.
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("spectrum")
    ps.add_argument("--indir", required=True)
    ps.add_argument("--outfile", default="spectrum.png")
    ps.set_defaults(func=cmd_spectrum)

    pf = sub.add_parser("field")
    pf.add_argument("--indir", required=True)
    pf.add_argument("--pattern", default="field_*.npy")
    pf.add_argument("--outfile", default="field.png")
    pf.set_defaults(func=cmd_field)

    args = p.parse_args()
    args.func(args)  # 선택된 서브커맨드에 연결된 함수를 실행


if __name__ == "__main__":
    main()
