# Мини-продукт с LLM-аналитикой

Веб-приложение на Streamlit для задания 3 по курсу «Аналитик данных». Пользователь загружает датасет, добавляет инструкцию или контекст, а LLM-агент через Gemini API сам вызывает локальный Python-интерпретатор, исследует данные, строит графики и возвращает отчет.

Агентная среда построена на `LangGraph`: это Python-фреймворк для stateful agent orchestration. По умолчанию LLM вызывается через пакет `g_api_view` из `https://github.com/brdchy/ApiClientG` и сервер `https://g-assistant-api.ru`. OpenRouter-адаптер оставлен как альтернативный провайдер.

## Что реализовано

- Веб-интерфейс для загрузки `csv`, `xlsx`, `xls`, `json`, `txt`.
- Поле для пользовательской инструкции: можно указать, на какие метрики, сегменты или гипотезы обратить внимание.
- Агентный цикл `LLM -> Python tool -> observation -> final report`.
- Gemini API через пакет `g_api_view`; OpenRouter можно включить через `LLM_PROVIDER=openrouter`.
- Генерация Markdown-отчета, ключевых метрик, инсайтов, ограничений и графиков.
- Защита от prompt-injection: инструкция и датасет считаются недоверенным контекстом, а инструкция отдельно проверяется LLM-классификатором перед запуском агента.

## Архитектура

```text
streamlit_app.py                 # точка входа Streamlit
analytics_agent/
  config.py                      # настройки и список моделей
  env.py                         # загрузка .env
  llm_client.py                  # фабрика выбора LLM-провайдера
  providers/                     # базовый LLM-клиент и провайдеры Gemini/OpenRouter
  data_loader.py                 # сохранение и профилирование датасета
  prompt_security.py             # LLM проверка prompt-injection
  code_sandbox.py                # Python tool для агента и safe helper save_current_plot
  agent.py                       # LangGraph-оркестрация агента
  report_writer.py               # сохранение отчета
  ui.py                          # Streamlit-интерфейс
input/kaggle_titanic.csv         # пример датасета
tests/                           # базовые тесты
```

## Установка

Нужен Python 3.10 или новее.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Настройка LLM API

Создайте `.env` по примеру `.env.example`:

```env
LLM_PROVIDER=gemini
GEMINI_SERVER_URL=https://g-assistant-api.ru
GEMINI_USER_ID=3838
GEMINI_PASSWORD=password
GEMINI_MODEL=Gemini 3.1 Flash Lite Preview
```

## Запуск

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

После запуска откройте адрес из терминала:

```text
http://localhost:8501
```

Для проверки можно загрузить `kaggle_titanic.csv` и написать инструкцию:

```text
Проанализируй выживаемость пассажиров Titanic по полу, классу билета и возрастным группам. Построй график с самым важным сравнением, найди заметные аномалии или пропуски в данных и напиши 3 главных инсайта.
```

## Как агент проводит анализ

1. Приложение сохраняет файл во временную папку и строит короткий технический профиль датасета.
2. LLM получает системные правила, профиль и пользовательский контекст.
3. `LangGraph` запускает граф состояний: `call_llm -> run_python_tool -> call_llm -> final`.
4. LLM возвращает JSON-команду `tool_call` с Python-кодом.
5. Приложение проверяет код через AST-фильтр и запускает его в отдельном процессе без доступа к ключам OpenRouter.
6. Результат выполнения возвращается LLM как observation.
7. После одного или нескольких шагов LLM формирует финальный отчет на русском языке.

Для графиков Python tool предоставляет безопасный helper `save_current_plot("chart.png")`. Он сохраняет текущую matplotlib-фигуру в разрешенную папку артефактов, после чего Streamlit автоматически показывает изображение в разделе «Графики».

Это важно для критерия оценки: LLM не просто перефразирует заранее посчитанные метрики из промпта, а сама выбирает вычисления и вызывает интерпретатор кода.

## Защита от prompt-injection

- Датасет и пользовательская инструкция явно помечены как недоверенные данные.
- Системный prompt запрещает раскрывать секреты, читать `.env`, использовать сеть и выполнять shell-команды.
- `LlmPromptInjectionGuard` отправляет инструкцию в настроенный LLM API как недоверенный текст и просит LLM вернуть JSON-классификацию с признаком вредоносности.
- `CodeSandbox` блокирует опасные импорты и вызовы: `os`, `subprocess`, `requests`, `open`, `eval`, `exec`, `__import__`, доступ к environment и операции записи/удаления файлов вне разрешенных графиков.
- Python tool запускается в отдельном процессе с урезанным набором переменных окружения.
