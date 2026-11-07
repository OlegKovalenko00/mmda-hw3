from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
COLORS = ['#742C45', '#3D6B62']
MUTED = '#817178'
WINE_CMAP = LinearSegmentedColormap.from_list('wine', ['#FEF4F2', '#D69BAD', '#742C45'])
plt.rcParams.update({
    'figure.facecolor': '#FBF6F3',
    'axes.facecolor': '#FFFDFC',
    'axes.edgecolor': '#DED0D5',
    'axes.titlecolor': '#742C45',
    'axes.labelcolor': '#30242A',
    'text.color': '#30242A',
    'xtick.color': '#55464D',
    'ytick.color': '#55464D',
    'grid.color': '#E7D9DC',
    'legend.facecolor': '#FFFDFC',
    'legend.edgecolor': '#DED0D5',
})


@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_probability(_model, features):
    return _model.predict_proba(features)[:, 1]


def read_features(upload, columns):
    frame = pd.read_csv(upload)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В файле нет строк с данными')
    frame = frame.loc[:, columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения')
    if not frame['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен содержать только 0 или 1')
    return frame


st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')
st.title('Подборка хороших вин')
st.caption('Оценка по лабораторным характеристикам вина')
model = load_model()
data = load_test()
features = data.drop(columns='target')
target = data['target']
import json
default_threshold = float(json.loads((BASE / 'config.json').read_text())['threshold'])
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, default_threshold,
                              step=0.001, format='%.3f', key='threshold')
st.sidebar.caption('Один порог применяется ко всем вкладкам')
single, evaluation, batch, interpretation = st.tabs(
    ['Оценить вино', 'Модельные метрики', 'CSV с винами', 'Что выучила модель'])

with single:
    labels = {'fixed acidity': 'Фиксированная кислотность', 'volatile acidity': 'Летучая кислотность',
              'citric acid': 'Лимонная кислота', 'residual sugar': 'Остаточный сахар',
              'chlorides': 'Хлориды', 'free sulfur dioxide': 'Свободный диоксид серы',
              'total sulfur dioxide': 'Общий диоксид серы', 'density': 'Плотность',
              'pH': 'pH', 'sulphates': 'Сульфаты', 'alcohol': 'Алкоголь'}
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(features.columns):
        with columns[index % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle('Красное вино', value=bool(features[feature].median())))
            else:
                minimum = float(features[feature].min())
                maximum = float(features[feature].max())
                values[feature] = st.number_input(labels.get(feature, feature),
                    min_value=minimum, max_value=maximum, value=float(features[feature].median()),
                    step=max((maximum - minimum) / 100, 0.0001), format='%.4f')
    sample = pd.DataFrame([values], columns=features.columns)
    probability = float(model.predict_proba(sample)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.success('Включить в подборку') if probability >= threshold else st.info('Не включать в подборку')

with evaluation:
    score = test_probability(model, features)
    predicted = score >= threshold
    matrix = metrics.confusion_matrix(target, predicted, labels=[0, 1])
    precision = metrics.precision_score(target, predicted, zero_division=0)
    recall = metrics.recall_score(target, predicted, zero_division=0)
    f1 = metrics.f1_score(target, predicted, zero_division=0)
    metric_columns = st.columns(3)
    for column, label, value in zip(metric_columns, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(label, f'{value:.3f}')
    st.caption('При увеличении порога подборка становится строже и обычно меньше. '
               'При уменьшении в неё попадает больше хороших вин, но могут добавиться посредственные')
    fig, ax = plt.subplots(figsize=(5, 3))
    metrics.ConfusionMatrixDisplay(matrix, display_labels=['Остальные', 'Хорошие']).plot(ax=ax, cmap=WINE_CMAP)
    ax.set(xlabel='Предсказанный класс', ylabel='Настоящий класс', title=f'Ошибки отбора при пороге {threshold:.3f}')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fpr, tpr, _ = metrics.roc_curve(target, score)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(target, score)
    tn, fp, fn, tp = matrix.ravel()
    axes[0].plot(fpr, tpr, color=COLORS[0], label=f'ROC-AUC={metrics.roc_auc_score(target, score):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color=MUTED)
    axes[0].scatter(fp / (fp + tn), recall, color=COLORS[1], label='Текущий порог', zorder=3)
    axes[1].plot(pr_recall, pr_precision, color=COLORS[0], label=f'PR-AUC={metrics.average_precision_score(target, score):.3f}')
    axes[1].axhline(target.mean(), ls='--', color=MUTED, label='Случайная модель')
    axes[1].scatter(recall, precision if predicted.any() else 1, color=COLORS[1], label='Текущий порог', zorder=3)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC-кривая на test')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая на test')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if not predicted.any():
        st.info('При этом пороге не отобрано ни одного вина')

with batch:
    st.write('Загрузите CSV с 12 признаками вина без столбца target')
    example = features.head(3).to_csv(index=False).encode('utf-8')
    st.download_button('Скачать пример CSV', example, file_name='example_wines.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками', type=['csv'])
    if uploaded is not None:
        try:
            batch_features = read_features(uploaded, features.columns.tolist())
            result = batch_features.copy()
            result['probability'] = model.predict_proba(batch_features)[:, 1]
            result['prediction'] = (result['probability'] >= threshold).astype(int)
            st.dataframe(result, hide_index=True)
            st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8'),
                               file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.error(f'Не удалось обработать файл: {error}')

with interpretation:
    estimator = model[-1]
    weights = estimator.w[1:] if hasattr(estimator, 'w') else estimator.coef_.ravel()
    coefficients = pd.Series(weights, index=features.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coefficients.index, coefficients.values,
            color=np.where(coefficients.values >= 0, COLORS[0], COLORS[1]))
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(xlabel='Коэффициент после масштабирования', ylabel='Признак',
           title='Какие характеристики связаны с хорошим вином')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино:', ', '.join(coefficients.tail(2).index[::-1]))
    st.write('Сильнее всего против:', ', '.join(coefficients.head(2).index))
    st.caption('Положительный коэффициент повышает оценку при увеличении признака на одно '
               'стандартное отклонение и неизменных остальных признаках. Это связь в данных, а не доказательство причины')
