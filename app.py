r"""실행: .venv\Scripts\python.exe -m streamlit run prototype/app.py"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from engine import (CLASS_KO, CLASS_NAMES, ROOT, FruitEngine, experiment_path,
                    model_signature, overview, result_zip)

st.set_page_config(page_title='과일 선화 CNN', page_icon='🍎', layout='wide')


@st.cache_resource(show_spinner=False)
def load_engine(run_path, seed, signature):
    return FruitEngine(run_path, seed)


def examples():
    folder = ROOT / 'results/lineart_v1_apple_banana_lemon_strawberry_pineapple/automatic_scenes'
    found = {}
    for path in sorted(folder.glob('*/settings.json')):
        settings = json.loads(path.read_text(encoding='utf-8'))
        source = settings.get('source_name')
        if source in ['strawberry.jpg', '12.jpg'] and (path.parent / 'photo.png').exists():
            found.setdefault('딸기 예시' if source == 'strawberry.jpg' else '사과와 단면 예시', path.parent / 'photo.png')
    return found


def render_result(result):
    count = len(result['objects'])
    if not count:
        st.warning('분석할 과일 영역을 찾지 못했습니다. 과일이 더 크게 보이는 사진으로 다시 시도해 주세요.')
        return
    st.subheader(f'분리한 과일 영역 {count}개')
    st.image(overview(result), caption='두 CNN 비교 시 사진 위에는 내부 특징 선화 CNN의 예측을 표시합니다.', width='stretch')
    display = result['table'][['과일 번호', '입력', '예측', '모델 점수', '사진 경계에 닿음']].copy()
    display['모델 점수'] = display['모델 점수'].map(lambda x: f'{x:.1%}')
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('모델 점수는 예측 확률값입니다. 실제 정답률이나 정확도를 뜻하지 않습니다. 영역이 잘못 분리되면 분류도 틀릴 수 있습니다.')
    idx = st.selectbox('자세히 볼 과일', range(count), format_func=lambda n: f'과일 #{n+1}', key='object_detail')
    obj = result['objects'][idx]
    mask = obj['mask']
    ys, xs = np.where(mask)
    cutout = result['rgb'].copy()
    cutout[~mask] = 255
    crop = cutout[ys.min():ys.max()+1, xs.min():xs.max()+1]
    cols = st.columns(3)
    for col, picture, caption in zip(cols, [crop, 255-obj['outline'], 255-obj['lineart']], ['분리한 과일', '외곽선', '내부 특징 선화']):
        with col:
            st.image(picture, caption=caption, width=260)
    rows = result['table'][result['table']['과일 번호'] == idx+1]
    scores = pd.DataFrame({row['입력']: [row[f'점수_{name}'] for name in CLASS_NAMES] for _, row in rows.iterrows()}, index=CLASS_KO)
    st.bar_chart(scores, horizontal=True, color=['#b8703e', '#285f4b'][:len(scores.columns)])
    st.download_button('선화·예측 결과 ZIP 다운로드', result_zip(result), 'fruit_result.zip', 'application/zip')


st.title('사진 속 과일, 선화로 알아보기')
st.write('사진에서 과일을 자동으로 분리하고, 컴퓨터가 만든 선화를 CNN으로 분류합니다.')
st.caption('분류 대상: 사과 · 바나나 · 레몬 · 딸기 · 파인애플')

try:
    run = experiment_path()
    seed, signature = model_signature(run)
except (OSError, ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header('분석 설정')
    mode = st.radio('사용할 CNN', ['내부 특징 선화 CNN', '외곽선 CNN', '두 CNN 비교'])
    rotation = st.checkbox('4방향 회전 평균', value=False, help='0°·90°·180°·270°의 예측 점수를 평균합니다. 항상 성능이 높아지지는 않습니다.')
    st.caption('기본값은 공개 테스트에서 성능이 가장 높았던 선화 CNN의 단일 예측입니다.')
    with st.expander('사용 모델과 처리 과정'):
        st.write('Grounding DINO 탐지 → SlimSAM 분리 → 선화 생성 → CNN 분류')
        st.code(f'{run.name}\nseed {seed}', language=None)
        st.write('노트북에서 비교 학습한 두 CNN을 사용합니다. 이 화면에서 재학습하지 않습니다.')

predict_tab, compare_tab = st.tabs(['사진 분석', '공개 테스트 비교'])
with predict_tab:
    samples = examples()
    source = st.selectbox('사진 선택 방법', ['내 사진 업로드', *samples])
    data = None
    if source == '내 사진 업로드':
        uploaded = st.file_uploader('분석할 사진을 선택하세요', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
        if uploaded is not None:
            data = uploaded.getvalue()
    else:
        data = samples[source].read_bytes()
    valid = False
    if data:
        try:
            with Image.open(io.BytesIO(data)) as image:
                preview = ImageOps.exif_transpose(image).convert('RGB')
                preview.thumbnail((1000, 650))
            st.image(preview, caption='입력 사진', width=500)
            valid = True
        except (OSError, ValueError, Image.DecompressionBombError):
            st.error('사진 파일을 읽지 못했습니다. JPG 또는 PNG 파일로 다시 선택해 주세요.')
    variants = ('outline', 'lineart') if mode == '두 CNN 비교' else (('outline',) if mode == '외곽선 CNN' else ('lineart',))
    key = (hashlib.sha256(data).hexdigest() if valid else None, variants, rotation, signature)
    if st.button('분석 시작', type='primary', disabled=not valid):
        st.session_state.pop('analysis_result', None)
        with st.spinner('과일을 분리하고 선화로 분류하고 있습니다. 첫 실행은 모델 준비 때문에 더 걸릴 수 있습니다.'):
            try:
                engine = load_engine(str(run), seed, signature)
                result = engine.analyze(data, variants, rotation)
                st.session_state.analysis_result = (key, result)
                st.session_state.pop('object_detail', None)
            except Exception as exc:
                st.error('분석을 완료하지 못했습니다. 아래 오류를 확인해 주세요.')
                with st.expander('오류 내용', expanded=True):
                    st.code(str(exc), language=None)
    saved = st.session_state.get('analysis_result')
    if saved and saved[0] == key:
        render_result(saved[1])
    elif valid:
        st.info('분석 시작을 누르면 과일별 선화와 예측을 확인할 수 있습니다.')
    st.caption('겹친 과일, 사진 밖으로 잘린 과일, 단면이나 학습 범위 밖 사물에서는 잘못 분리하거나 오분류할 수 있습니다.')

with compare_tab:
    st.subheader('동일한 테스트 사진에서 두 CNN 비교')
    metrics = run / 'report/all_metrics.csv'
    if metrics.exists():
        raw = pd.read_csv(metrics)
        table = raw[['seed', 'input', 'inference', 'test_count', 'accuracy', 'macro_f1']].copy()
        table.columns = ['seed', '입력', '예측 방식', '테스트 수', '정확도', 'Macro F1']
        table['입력'] = table['입력'].replace({'outline': '외곽선', 'lineart': '내부 특징 선화'})
        table['예측 방식'] = table['예측 방식'].replace({'single_0deg': '0° 단일 예측', 'rotation_mean_4': '4방향 평균'})
        table['정확도'] = table['정확도'].map(lambda x: f'{x:.1%}')
        table['Macro F1'] = table['Macro F1'].map(lambda x: f'{x:.4f}')
        st.dataframe(table, hide_index=True, width='stretch')
        primary = raw[raw.inference == 'single_0deg'].groupby('input')[['accuracy', 'macro_f1']].mean()
        primary.index = primary.index.map({'outline': '외곽선', 'lineart': '내부 특징 선화'})
        st.bar_chart(primary.rename(columns={'accuracy': '정확도', 'macro_f1': 'Macro F1'}))
        st.download_button('비교표 CSV 다운로드', metrics.read_bytes(), 'cnn_comparison.csv', 'text/csv')
        st.caption('학습 조건을 맞춘 공개 데이터 평가입니다. 업로드한 사진의 정확도나 자동 분리 성능을 의미하지 않습니다.')
    else:
        st.info('노트북의 4-B를 실행하면 비교 결과가 여기에 나타납니다.')
