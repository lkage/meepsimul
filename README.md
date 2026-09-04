# THz pinch-harmonics — Meep reproduction

Meep/FDTD reproduction of the simulation in:

> H.-R. Park, Y.-M. Bahk, J. H. Choe, S. Han, S. S. Choi, K. J. Ahn,
> N. Park, Q.-H. Park, and D.-S. Kim, "Terahertz pinch harmonics enabled
> by single nano rods," *Opt. Express* **19**(24), 24775–24781 (2011).

## 로컬 환경에서 시작하기 (git clone 이후 실행 순서)

아래는 이 저장소를 로컬 머신에 `git clone`으로 받아온 뒤, 실제로 스모크
테스트를 돌려보기까지 필요한 명령어를 순서대로 정리한 것이다. (Linux /
macOS 기준. Windows는 WSL 사용을 권장.)

### 0) 사전 준비물

- **conda(또는 mamba)**: pymeep은 PyPI(`pip install meep`)로는 제대로 설치되지
  않고 **conda-forge 채널로만** 정상 배포된다. conda가 없다면 먼저
  [Miniforge](https://github.com/conda-forge/miniforge)를 설치할 것
  (conda-forge가 기본 채널로 설정되어 있어 가장 간단하다).
- **git**

### 1) 저장소 클론

```bash
git clone https://github.com/lkage/meepsimul.git
cd meepsimul
```

(이미 이 브랜치 상태를 로컬에 갖고 있다면 `git clone` 대신 `git pull`로 최신화만 해도 된다.)

### 2) conda 환경 생성 및 활성화

저장소에 포함된 `environment.yml`이 pymeep(MPI 병렬 빌드) + numpy/matplotlib/scipy를
한 번에 설치해 준다.

```bash
conda env create -f environment.yml
conda activate pinch-harmonics
```

이미 pymeep이 설치된 별도의 conda 환경을 쓰고 싶다면, 그 환경을 활성화한
뒤 나머지 파이썬 의존성만 추가로 설치해도 된다:

```bash
conda activate <기존-pymeep-환경-이름>
pip install -r requirements.txt   # numpy, matplotlib, scipy
```

### 3) 설치 확인

```bash
python3 -c "import meep as mp; print(mp.__version__)"
mpirun --version
```

두 명령 모두 에러 없이 버전이 출력되면 준비 완료.

### 4) 스크립트 문법/실행 확인 (빠른 스모크 테스트)

먼저 각 스크립트의 `--help`가 정상적으로 뜨는지 확인:

```bash
python3 simulate.py --help
python3 analyze.py --help
```

그다음, 기본 저해상도(`resolution=60`) 설정으로 전체 파이프라인을 한 번
가볍게 돌려본다. 이 단계는 **논문 수준의 정확한 결과가 아니라 "코드가
제대로 도는지" 확인용**이며 노트북에서도 수 분~수십 분 내로 끝나는
정도의 규모다:

```bash
chmod +x run_all.sh          # 실행 권한이 없다는 오류가 나올 때만 필요
bash run_all.sh
```

정상적으로 끝나면 다음 파일들이 생성된다:

```
results/pos12/spectrum.png   # Fig. 2(b) 스타일 투과율 스펙트럼
results/pos11/spectrum.png   # Fig. 3(b) 스타일 투과율 스펙트럼
results/pos12/flux_*.npz     # 원본 플럭스 데이터
results/pos11/flux_*.npz
```

콘솔에는 `find_peaks`가 찾아낸 후보 공진 주파수 목록도 함께 출력되니,
다음 단계(필드 맵 추출)에서 사용할 주파수 값을 여기서 읽으면 된다.

### 5) 본격적인(신뢰할 만한) 재현 실행

스모크 테스트가 문제없이 끝났다면, 해상도를 크게 올리고 MPI 프로세스
수도 머신 사양에 맞게 늘려서 실행한다. **20 nm급 구조를 제대로 표현하려면
해상도가 수백(1/μm) 수준은 되어야 하며, 이 경우 시간/메모리 비용이
크게 증가한다** (README의 "Honesty / caveats" 절 참고):

```bash
RESOLUTION=300 NP=32 bash run_all.sh
```

### 6) |Ex|² 필드 맵(Fig. 2(c)/3(c)) 재현

`run_all.sh` 실행 로그에 출력된 1차/3차 하모닉 후보 주파수를 `<f1>`,
`<3f1>` 자리에 넣어 아래처럼 직접 실행한다 (예시는 `run_all.sh` 마지막
안내 문구에도 동일하게 출력됨):

```bash
mpirun -np $NP python3 simulate.py --mode field --rod none \
    --freqs <f1>,<3f1> --outdir results/pos12
mpirun -np $NP python3 simulate.py --mode field --rod large --position 1:2 \
    --freqs <f1>,<3f1> --outdir results/pos12
mpirun -np $NP python3 simulate.py --mode field --rod small --position 1:2 \
    --freqs <f1>,<3f1> --outdir results/pos12

python3 analyze.py field --indir results/pos12 --pattern 'field_*.npy' \
    --outfile results/pos12/fields.png
```

### 자주 겪을 수 있는 문제

- `conda env create` 단계가 오래 걸리거나 멈추는 경우: conda 대신
  `mamba env create -f environment.yml`을 쓰면 의존성 해결 속도가 훨씬
  빠르다 (Miniforge에는 mamba가 기본 포함).
- `mpirun: command not found`: `environment.yml`의 `pymeep=*=mpi_mpich_*`
  빌드가 아니라 serial(비-MPI) 빌드가 설치된 경우일 수 있다. 이 경우
  `mpirun -np N` 없이 `python3 simulate.py ...`처럼 단일 프로세스로
  실행하면 된다 (다만 대규모 해상도에서는 매우 느려진다).
- 메모리 부족(OOM)으로 죽는 경우: `--resolution`을 낮추거나, `NP`(랭크 수)를
  늘려 프로세스당 메모리 사용량을 분산시킨다.

## What the paper's FDTD model is

Section 2 ("To simulate pinch harmonics by Pt nano rods, we perform
three-dimensional finite-difference time-domain (FDTD) analysis using
Drude model...") describes:

- a single rectangular slot ("hole"), length **l1 = 10 μm**, width
  **w1 = 20 nm**, cut through a **20-nm-thick gold film** — a scaled-down
  stand-in for the real device's 300 μm × 120 nm slot antenna, chosen so
  the sub-skin-depth physics is preserved while the simulation stays
  tractable;
- a normally-incident, broadband THz pulse polarized along the slot's
  *short* axis (p-polarization, "perpendicular to the long axis of the
  rectangle");
- a single Pt nano rod sitting on top of the slot, oriented across it,
  with width **s = 60 nm** ("large") or **s = 20 nm** ("small") along the
  slot's long axis, placed either at the **1:2** position (Fig. 2) or the
  **1:1** midpoint (Fig. 3);
- reading out (a) the transmittance spectrum and (b) |Ex|² field maps
  just behind the exit face of the slot, at the fundamental and 3rd
  harmonic resonance.

This repo implements exactly that structure and those two measurements.
It does **not** attempt to reproduce the full-size experimental device
(300 μm slot, 120 nm width) — the paper itself only simulates the scaled
structure above.

## Files

| file | purpose |
|---|---|
| `materials.py` | metal media (see caveat below) |
| `geometry.py` | builds the slot + rod geometry as Meep objects |
| `simulate.py` | CLI driver: `--mode spectrum` (Fig. 2b/3b) or `--mode field` (Fig. 2c/3c) |
| `analyze.py` | loads simulate.py output, makes overlay/field plots, reports resonance peaks |
| `run_all.sh` | drives all the runs needed for Figs. 2 and 3 |
| `environment.yml` | conda-forge env spec (pymeep is not on PyPI) |

## Coordinate convention

`z` = propagation direction of the incident pulse; `x` = E-field
polarization = slot **width** direction (w1); `y` = slot **length**
direction (l1), i.e. the axis the rod's position ("1:2", "1:1") is
measured along. This matches the `Ex, Hy, kz` labels in Fig. 1(a) of the
paper.

## Honesty / caveats — please read before trusting numbers

This code was written and reviewed for **API correctness** against
Meep's documented Python interface, but **was not executed** in the
session that produced it: pymeep is a conda-forge-only package (its
PyPI listing is unrelated / fails to build) and installing a full
conda-forge pymeep stack plus running a meaningfully-resolved 3D FDTD job
was outside what was practical there. So:

1. **Not numerically validated.** No run of this code has actually been
   performed to confirm it reproduces the paper's curves. Please treat
   it as a carefully-written starting point, and sanity-check your first
   results (e.g. against the qualitative claims in the paper: a bare
   slot shows a dominant fundamental and weaker 3rd harmonic; a 600 nm
   rod at 1:2 splits/shifts the fundamental; a 250 nm rod at 1:2 makes
   the 3rd harmonic dominant; either rod at the 1:1 midpoint suppresses
   the fundamental, and the 20 nm-scale rod there additionally kills the
   symmetric n=2 mode).
2. **Metal model.** `materials.py` uses Meep's bundled
   `meep.materials.Au` / `.Pt` (a 6-pole Lorentz-Drude fit to Rakic et
   al., Appl. Opt. 37, 5271 (1998)), fitted against ~0.2–12 μm
   (near-IR–UV) optical data — *not* against THz/far-IR data. The paper
   itself uses a plain Drude model citing Ordal et al., Appl. Opt. 22,
   1099 (1983) (its Ref. [22]). In the simulated band the interband
   (Lorentzian) poles of the Rakic fit contribute negligibly and the
   free-electron Drude pole dominates, so this is a reasonable
   extrapolation and is *exactly* what ships with Meep (nothing
   hand-transcribed) — but it is still an extrapolation, and the two
   models' Drude parameters are not identical. For a quantitatively
   exact reproduction, look up Ordal's tabulated ω_p/ω_τ for Au and Pt
   yourself and pass them via `--custom-drude-film f_p,gamma` /
   `--custom-drude-rod f_p,gamma` (values in Meep frequency units, i.e.
   1/λ[μm], with the length unit a = 1 μm used throughout this project;
   1 cm⁻¹ = 1e-4 of that unit).
3. **Rod thickness.** The paper states the FDTD rod *widths* (60 nm /
   20 nm) but not its thickness for the scaled model. `--t2` defaults to
   20 nm (same as the film) as a reasonable placeholder — adjust it if
   you have a better basis for it.
4. **Rod length (l2).** Likewise unspecified for the scaled model;
   `--l2` defaults to 300 nm (must be > w1 = 20 nm so the rod actually
   bridges the slot, plus margin to sit on the film on both sides) —
   arbitrary and adjustable.
5. **Normalization.** Transmittance is computed as
   `P_trans(f; with structure) / P_inc(f; empty cell, same source and
   monitor-plane geometry)`, run as two independent Meep runs (`--empty`
   for the reference), which is the standard way to turn a flux spectrum
   into a transmittance spectrum and matches a THz-TDS sample/reference
   scan. It is not derived from the paper's own (unpublished) FDTD
   normalization procedure.
6. **Resolution / cost.** To resolve a 20 nm slot/rod with even a modest
   ~8–10 grid points needs `resolution` on the order of a few hundred
   (1/μm) over a domain spanning ~10+ μm — a genuinely large 3D job.
   `simulate.py`'s default (`resolution=60`) is a fast, low-fidelity
   *smoke test* to check the script runs and the geometry looks right,
   not a publication-quality setting. Expect to need MPI (`mpirun -np
   N`) and a machine with real memory/core budget for a faithful run;
   consider Meep's chunk/load-balancing options if scaling to many
   ranks.
7. **PML thickness.** The source pulse's bandwidth (default `--fcen
   0.09 --df 0.16`, i.e. wavelengths roughly 3–300 μm in this unit
   system) is very broad relative to the sub-μm structure, so the PML
   (`--dpml`, default 1 μm) will absorb the low-frequency tail somewhat
   imperfectly. If you see reference-run transmittance deviating
   noticeably from 1 outside the frequency band you actually care about,
   narrow `--fcen/--df` to your region of interest and/or increase
   `--dpml`.

## Installing pymeep / Running

See "로컬 환경에서 시작하기 (git clone 이후 실행 순서)" near the top of
this file for the full step-by-step (clone, env setup, smoke test,
production run, field-map extraction, troubleshooting).

Quick reference:

```
conda env create -f environment.yml     # or: conda install -c conda-forge pymeep=*=mpi_mpich_*
conda activate pinch-harmonics
bash run_all.sh                         # fast smoke test at resolution=60
RESOLUTION=300 NP=32 bash run_all.sh    # a much more serious run
```

or drive `simulate.py` directly — see `python3 simulate.py --help` and
the module docstring at the top of the file for example invocations of
`--mode spectrum` and `--mode field`.
