# STEP Face Mapping POC (python + OpenCascade)

FreeCAD API 없이 STEP 파일을 OpenCascade 바인딩으로 파싱하고, Face 단위 매핑 리포트 + 클릭 가능한 웹 뷰어를 제공하는 POC입니다.

## 기능
- STEP(.stp/.step) 로딩 (`cadquery-ocp`의 `OCP` 우선, 일부 환경에서 `pythonocc-core` 폴백)
- Face별 고정 ID 부여: `Face1..FaceN`
- Face 메타데이터 추출
  - `face_id`, `surface_type`, `area`, `center_of_mass`
  - `bbox_min/max`, `edge_count`
  - `display_label` (타입/치수/면적/엣지수 기반 라벨)
  - `edge_type_counts`, `dominant_edge_type` (Line/Circle/BSplineCurve 등)
  - 원통/원뿔/구/토러스 반지름/축 정보
  - 옵션 normal(midpoint)
- 전역 메타데이터
  - solids/shells/faces/edges/vertices count
  - 글로벌 bounding box
  - total surface area / volume(가능한 경우)
  - STEP header(best-effort): schema/name/description/units_raw
  - STEP 엔티티 이름 통계: `ADVANCED_FACE`, `EDGE_CURVE`, `MANIFOLD_SOLID_BREP`, `PRODUCT`
- 3D 출력
  - Face별로 분리된 OBJ (`out/model.obj`, 각 오브젝트 이름이 `FaceN`)
- 웹 뷰어(`/viewer/index.html`)
  - 초보자용 용어 설명(한글) + STEP/OCCT 공식 타입명 병기
  - Face 목록 표시 (`FaceID + 한글 surface명 + 경계선/면적 요약`)
  - Surface 타입별 고대비 색상/범례
  - 타입 필터 + 검색
  - 정렬(번호순/면적순), 이전/다음 Face 이동, 선택 면 확대
  - 클릭 피킹/리스트 선택 시 외곽선 강조 + 비선택 면 디밍
  - `report.json` 기반 메타데이터 표시
  - STEP 이름 필드 존재 여부 표시

## 저장소 구조
```text
src/
  main.py
  step_loader.py
  topology_extract.py
  geometry_report.py
  tessellate_export.py
viewer/
  index.html
  serve.py
out/
```

## 1) venv 기반 실행
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.main --input path/to/model.step --out out/
```

출력 파일:
- `out/report_<input-file-stem>.json`
- `out/faces_<input-file-stem>.csv`
- `out/model_<input-file-stem>.obj`

## 2) 뷰어 실행
브라우저 `file://`로 직접 열면 fetch/CORS 이슈가 날 수 있어 로컬 서버 권장:

```bash
python viewer/serve.py --port 8000
```

브라우저 접속:
- `http://127.0.0.1:8000/viewer/`

기본 로드 경로:
- 모델/리포트는 뷰어 상단 드롭다운에서 선택 (`out` 폴더 내 `model*.obj` + `report*.json` 쌍 자동 인식)

필요하면 URL 파라미터 지정 가능:
- `http://127.0.0.1:8000/viewer/?model=../out/model.obj&report=../out/report.json`

정적 호스팅용 모델 목록 파일 생성:
```bash
python viewer/build_models_manifest.py
```
생성 파일: `viewer/models.json`

## 3) Docker (선택)
```bash
docker build -t step-face-poc .

docker run --rm \
  -v "$PWD:/work" \
  step-face-poc \
  --input /work/path/to/model.step \
  --out /work/out
```

## 설치 이슈 및 대안
일부 Linux 환경에서 `pythonocc-core` pip wheel이 없을 수 있습니다. 이 프로젝트는 기본적으로 `cadquery-ocp`를 사용합니다.

필요 시 conda 대안:
```bash
conda create -n step-poc python=3.11 -y
conda activate step-poc
conda install -c conda-forge pythonocc-core -y
pip install -r requirements.txt
```

## CLI
```bash
python -m src.main --input path/to/model.step --out out/
```
옵션:
- `--output-stem` (기본: 입력 파일 stem)
- `--linear-deflection` (기본 0.2)
- `--angular-deflection` (기본 0.3)
- `--log-level` (기본 INFO)

## 무료 호스팅 제안
1. Render (가장 간단)
- Python 서버(`viewer/serve.py`) 그대로 사용 가능
- 장점: `/api/models` 자동 동작, 별도 정적 설정 불필요
- 단점: free tier 슬립/콜드스타트

2. GitHub Pages / Netlify / Cloudflare Pages (정적)
- `viewer/models.json`을 기준으로 모델 목록 로드
- 배포 전 `python viewer/build_models_manifest.py` 실행 필요
- 기본 `.gitignore`는 `out/`, `out_v2/` 결과물을 제외하므로, 정적 배포 시에는 결과물 포함 전략이 필요
- 장점: 무료/빠름/설정 단순
- 단점: 서버 API 없음, 모델 목록은 매니페스트 재생성 필요

## Vercel 배포 (레포 연결)
이 저장소는 `vercel.json`이 포함되어 있어 레포 연결 시 기본 동작이 설정됩니다.

- `/` 접속 시 `/viewer/`로 리라이트
- `/api/models` 요청 시 `/viewer/models.json` 사용
- 빌드 시 `python3 viewer/build_models_manifest.py` 실행

배포 전에 한 번 실행:
```bash
python viewer/build_models_manifest.py
```

레포에 포함할 파일:
- `viewer/index.html`
- `viewer/models.json`
- `vercel.json`
- `out_v2/model*.obj`
- `out_v2/report*.json`

`.gitignore`는 `out_v2`에서 위 파일(`model*.obj`, `report*.json`)만 추적하도록 설정되어 있습니다.

Vercel 설정(Import Git Repository):
- Framework Preset: `Other`
- Build Command: `python3 viewer/build_models_manifest.py` (vercel.json 기본값 사용 가능)
- Output Directory: 비워두기(루트 정적 파일 사용)

`cadquery-ocp`는 현재 CPython 3.14(`cp314`) 휠이 없어 Vercel 기본 Python 의존성 설치에서 실패할 수 있습니다.
이 저장소는 `vercel.json`의 `installCommand`를 no-op으로 설정해 정적 뷰어 배포 시 해당 오류를 회피합니다.

## GitHub 업로드 준비
이 저장소는 `out/`, `out_v2/`를 결과물 폴더로 사용하므로, Git에는 폴더만 유지하고 내부 생성 파일은 기본적으로 제외합니다.

초기 업로드 절차:
```bash
git init
git add .
git commit -m "init: STEP parser + viewer"
git branch -M main
git remote add origin https://github.com/<your-id>/<repo>.git
git push -u origin main
```

결과물은 필요할 때 재생성:
```bash
python -m src.main --input path/to/model.step --out out_v2/
```

## 참고
- glTF가 필요하면 OBJ->glTF 변환 단계(예: assimp/blender/obj2gltf)를 추가하면 됩니다.
- 현재 POC는 Face별 선택/식별을 핵심 목표로 OBJ 분리 방식을 채택했습니다.
