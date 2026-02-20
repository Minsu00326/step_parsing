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
