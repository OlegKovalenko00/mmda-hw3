# Подборка вин

Проверено на Python 3.12.14. Перейдите в папку wine_app и выполните:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Файлы model.joblib, config.json, test.csv и logreg_model.py должны лежать рядом с app.py. Модель уже обучена, приложение только рассчитывает прогнозы.

cache_resource сохраняет один объект модели между перезапусками. cache_data хранит таблицу и вероятности, возвращая копию данных. Без декоратора joblib.load заново читал бы модель при каждом действии пользователя.
