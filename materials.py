"""THz pinch-harmonics 재현에 쓰이는 금속(Au, Pt) 재료 모델.

[배경: 왜 "Drude 모델"인가?]
논문(Park et al., Opt. Express 19, 24775 (2011))의 FDTD 시뮬레이션은
금(Au)과 백금(Pt)을 "Drude 모델"로 기술한다고 밝히고 있다 (Sec. 2).
Drude 모델은 금속 내부의 자유전자(free electron)를 감쇠 진동자로 취급하는
가장 단순한 금속 유전율 모델로, 다음 형태를 가진다:

    eps(w) = eps_inf - w_p^2 / (w^2 + i*w*gamma)

  - w      : 각주파수 (여기서는 Meep 단위, 아래 "단위계" 설명 참고)
  - w_p    : 플라즈마 주파수 (금속 내 자유전자 밀도에 비례, 클수록 금속성이 강함)
  - gamma  : 전자 충돌(scattering) 감쇠율. 클수록 손실(loss)이 큼
  - eps_inf: 아주 높은 주파수에서의 유전율 (보통 1로 둠)

THz(0.1~수 THz) 대역에서는 금/백금 모두 광자 에너지가 매우 작아서
(meV 수준) 띠간전이(interband transition, 보통 eV 수준에서 발생)는
거의 기여하지 않고, 순수 자유전자(Drude) 반응이 지배적이다. 그 결과
THz 영역에서 금속은 "표피 깊이(skin depth)"가 매우 얇은 거의 완전한
도체처럼 행동한다 -- 이 논문의 "skin depth보다 얇은 나노로드가 공진을
억제한다"는 핵심 물리도 바로 이 성질에서 나온다.

[숫자를 직접 베끼지 않은 이유]
Drude 계수(w_p, gamma)를 이 파일에 손으로 옮겨 적으면 오타/기억 오류로
값이 틀릴 위험이 있다. 그래서 기본 경로는 Meep 패키지에 이미 내장되어
검증된 재료 라이브러리(meep.materials.Au / meep.materials.Pt)를 그대로
가져다 쓴다. 이 라이브러리는 Rakic et al., Appl. Opt. 37, 5271 (1998)의
6중극(6-pole) Lorentz-Drude 피팅을 구현한 것으로, 자유전자(Drude) 극 1개
+ 띠간전이를 나타내는 Lorentz 극 5개로 구성된다. 즉 "Drude 모델"보다
조금 더 정교한 상위호환 모델이며, 저주파(THz) 극한에서는 Drude 극만
남기 때문에 논문이 쓴 순수 Drude 모델과 사실상 동일하게 거동한다.

[중요한 주의사항 -- 반드시 읽을 것]
Meep의 Au/Pt 피팅은 약 0.2~12 um 파장(근적외선~자외선) 광학 데이터에
맞춰 피팅된 것이지, 원적외선/THz 데이터로 직접 검증된 것이 아니다.
논문 자체는 Ordal et al., Appl. Opt. 22, 1099 (1983) (논문의 참고문헌
[22])의 Drude 계수를 인용한다. 위에서 설명했듯 THz 대역에서는 Rakic
피팅의 Lorentz(띠간전이) 극들의 기여가 거의 0에 가깝고 자유전자
Drude 극이 지배적이므로, meep.materials.Au/Pt를 그대로 쓰는 것은
물리적으로 합리적인 외삽(extrapolation)이며 "검증 가능"하다는 장점이
있다 (제가 임의로 만든 숫자가 아니라 Meep 배포판에 실제로 들어있는
값 그대로이기 때문). 하지만 여전히 외삽이지 THz 대역에서 직접 검증된
피팅은 아니다. 정량적으로 논문과 완전히 동일한 결과를 원한다면 Ordal
논문의 실제 w_p/gamma 표를 직접 찾아서 --custom-drude-film /
--custom-drude-rod 옵션(simulate.py --help 참고)으로 넘기면, 순수
단일극 Drude 매질을 그 값으로 직접 구성한다.

[Meep 단위계에 대해]
이 프로젝트 전체는 Meep의 관례대로 "길이 단위 a = 1 um"를 사용한다.
Meep은 무차원화된 단위계를 쓰는데, 주파수 단위는 c/a 이므로
    f_meep = a / lambda = 1 / lambda[um]
가 된다. 즉 어떤 주파수를 Meep 단위로 표현한 값은 "그 파장을 um으로
쟀을 때의 역수"와 같다. 예를 들어 lambda = 10 um 이면 f_meep = 0.1 이다.
"""

import meep as mp
import meep.materials as mm

# 이 딕셔너리는 문자열 이름("Au","Pt")을 Meep에 내장된 mp.Medium 객체로
# 매핑한다. mm.Au / mm.Pt는 meep 패키지 설치 시 함께 딸려오는
# meep/materials.py 모듈이 정의해 둔 상수(모듈 import 시점에 이미
# Lorentz-Drude 극들이 채워진 mp.Medium 인스턴스)이다.
_BUILTIN = {"Au": mm.Au, "Pt": mm.Pt}


def get_metal(name, custom_drude=None):
    """이름으로 지정된 금속의 mp.Medium(유전체 매질) 객체를 반환한다.

    Parameters
    ----------
    name : "Au" 또는 "Pt". custom_drude가 주어지면 이름은 로그/구분용으로만
        쓰이고 실제 매질은 custom_drude 값으로 만든다.
    custom_drude : (f_p, gamma) 튜플, 둘 다 Meep 주파수 단위(1/um, a=1um
        기준). 주어지면 라이브러리 재질 대신 아래처럼 순수 단일극 Drude
        매질을 직접 구성한다:

            eps(w) = 1 - f_p^2 / (w^2 + i*w*gamma)

        예) Ordal 표에서 금의 w_p = 72800 cm^-1, gamma = 215 cm^-1 라면
        (1 cm^-1 = 1e-4 [1/um] 이므로) f_p=7.28, gamma=0.0215 로 변환해서
        넘기면 된다.
    """
    if custom_drude is not None:
        f_p, gamma = custom_drude
        # mp.DrudeSusceptibility(frequency, gamma, sigma) 는 Meep이 제공하는
        # 분산성(dispersive) 감수율(susceptibility) 항으로,
        #   chi(w) = sigma * frequency^2 / (-w^2 - i*w*gamma)
        # 를 구현한다. 즉 frequency 인자는 "공진 주파수"가 아니라 Drude 모델의
        # 플라즈마 주파수 w_p 역할을 한다 (sigma=1일 때
        # eps(w) = eps_inf + chi(w) = 1 - w_p^2/(w^2+i*w*gamma) 가 되어
        # 위에서 설명한 표준 Drude 식과 정확히 일치).
        # E_susceptibilities=[...] 는 이 감수율 항을 전기장(E) 성분에 붙인다는
        # 뜻이며, epsilon=1.0 은 eps_inf (아주 높은 주파수에서의 유전율)이다.
        return mp.Medium(
            epsilon=1.0,
            E_susceptibilities=[mp.DrudeSusceptibility(frequency=f_p, gamma=gamma, sigma=1)],
        )
    if name not in _BUILTIN:
        raise ValueError(f"no built-in metal '{name}'; supported: {list(_BUILTIN)} "
                          f"(or pass --custom-drude-* with your own f_p,gamma)")
    return _BUILTIN[name]
