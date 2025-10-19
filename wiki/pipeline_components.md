# Текущая архитектура пайплайна

## Цель

Справочник по ключевым участкам текущего пайплайна (`pipeline.py`, `scene_builder.py`, `prompting.py`, `scorer.py`, `storage.py`, `config.py`) с акцентом на зависимости и ответственность. Документ помогает команде сопоставить существующие функции с целевыми слоями (`scene`, `caption`, `image`, `scoring`, `db`).

## Оркестратор `pipeline.py`

| Область | Детали |
| --- | --- |
| Основные типы | `PipelineServices`, `PromptWorkflow`, функции `run_plain` и `run_evolve` управляют основной логикой запуска. |
| Входы | `PipelineConfig` из `config.py`, внешние сервисы Google Imagen (`google.genai`), Ollama API, каталоги контента, настройки скоринга. |
| Зависимости | Инициализация каталога (`Catalog`), библиотеки персонажей (`CharacterLibrary`), конструктора сцен (`SceneBuilder`), композитора промптов (`PromptComposer`), скорера (`DualScorer`), кеша эмбеддингов (`EmbeddingCache`), обратной связи (`StyleFeedback`), логгера/хранилища (`PromptLogger`, `ArtifactWriter`). |
| Ответственность | Конфигурация сервисов, цикл генерации сцен → подписи → изображения → скоринг, взаимодействие с Ollama/Imagen, запись артефактов и метрик. |
| Наблюдения | Оркестратор знает о внутренних структурах всех слоёв (генетический алгоритм, сцены, скоринг, сторидж), что затрудняет выделение независимых подсистем. |

### Внутренний workflow

1. `_prepare_services` собирает и передаёт в `PipelineServices` все зависимости, в том числе побочные эффекты (I/O, загрузка весов, создание клиентов). 【F:imagen_lab/pipeline.py†L66-L191】
2. `run_plain` запускает последовательность *scene → caption → image → scoring → storage* в цикле, управляет Ollama и Imagen, а также отправляет метрики обратной связи. 【F:imagen_lab/pipeline.py†L200-L362】【F:imagen_lab/pipeline.py†L363-L470】
3. `run_evolve` расширяет тот же цикл генетическим алгоритмом, но повторно использует объекты `SceneBuilder`, `PromptWorkflow`, `DualScorer`, что усиливает связанность. 【F:imagen_lab/pipeline.py†L471-L798】

## `SceneBuilder`

| Область | Детали |
| --- | --- |
| Основные типы | `SceneStruct`, `PromptPayload`, методы `build_scene`, `rebuild_from_genes`. |
| Входы | Каталог JSON (`Catalog`), уровни SFW/temperature, bias из `StyleFeedback`, кастомные каталоги персонажей. |
| Зависимости | Инфраструктурные утилиты (случайные выборы из `randomization.py`), библиотеки персонажей, каталоги гардероба и реквизита. |
| Ответственность | Собирает структурированное описание сцены: выбранные опции каталога, итоговый pre-prompt (`PromptPayload`), подсказки для обратной связи и статистику генов. |
| Наблюдения | Логика выбора wardrobe/props смешана с обработкой персонажей и bias; builder возвращает payload готовый для Ollama, то есть слой `scene` уже знает про требования `caption`. 【F:imagen_lab/scene_builder.py†L14-L272】【F:imagen_lab/scene_builder.py†L273-L402】【F:imagen_lab/scene_builder.py†L403-L476】

## `PromptComposer`

| Область | Детали |
| --- | --- |
| Основные типы | Класс `PromptComposer`, функции `ollama_generate`, `append_required_terms`. |
| Входы | `StyleGuide`, список обязательных терминов, JSON payload сцены. |
| Зависимости | HTTP-клиент `requests`, внутренняя утилита `StyleGuide`, глобальные функции из того же модуля. |
| Ответственность | Формирование системного промпта для Ollama, контроль обязательных терминов, enforcement одного шага через Ollama, финальная пост-обработка текста перед Imagen. |
| Наблюдения | Класс напрямую вызывает Ollama API, управляет enforcement и финальной склейкой промпта, тем самым совмещая слои `caption` (подготовка текста) и `image` (учёт ограничений Imagen). 【F:imagen_lab/prompting.py†L9-L126】【F:imagen_lab/prompting.py†L127-L222】

## `DualScorer`

| Область | Детали |
| --- | --- |
| Основные типы | `DualScorer`, вспомогательные структуры `ScoreResult`, `ScoredImage`. |
| Входы | Пути к сохранённым изображениям, весовые профили стиля, настройки CLIP/opennsfw, история эмбеддингов. |
| Зависимости | Множество ML-библиотек (`torch`, `open_clip`, `opennsfw2`, `skimage`, `PIL`), внутренняя композиция стиля (`StyleMixer`), анализ поз/композиции. |
| Ответственность | Вычисление nsfw/style метрик, сбор микрометрик (pose, composition, specular), агрегация в итоговый fitness, запись результатов в SQLite и JSONL, обновление весов и кешей. |
| Наблюдения | Класс соединяет вычисление, адаптацию весов и запись в БД; предоставляет API `score_and_save`, который возвращает `ScoredImage` и обновляет веса в том же вызове. 【F:scorer.py†L544-L663】【F:scorer.py†L664-L782】

## `storage.py`

| Область | Детали |
| --- | --- |
| Основные типы | `ArtifactWriter`, `PromptLogger`, `save_and_score`. |
| Входы | Ответ Imagen API (`response.generated_images`), финальный промпт, сцена, сервисы логирования и скоринга. |
| Зависимости | Локальная файловая система, SQLite (`PromptLogger`), numpy для метрик, кеш эмбеддингов, сериализация JSON. |
| Ответственность | Сохранение изображений и сайд-каров, запись EXIF, логов промптов, агрегация метрик батча/истории, обновление EmbeddingCache. |
| Наблюдения | Функция `save_and_score` содержит логику подсчёта средних метрик и enrichment JSON, что относится к слою `scoring`, а также дергает `PromptLogger` (слой `db`). 【F:imagen_lab/storage.py†L1-L204】【F:imagen_lab/storage.py†L205-L344】

## Конфигурация (`config.py` и `config.yaml`)

| Область | Детали |
| --- | --- |
| Основные типы | `PipelineConfig` и вложенные dataclass-конфиги (`PathsConfig`, `PromptConfig`, `ScoringConfig`, др.). |
| Входы | YAML-конфиг, пути к каталогам, параметры Ollama/Imagen, веса скоринга, настройки GA. |
| Зависимости | `yaml.safe_load`, pathlib. |
| Ответственность | Валидация и нормализация пользовательских настроек, предоставление typed-конфига для `pipeline.py`. |
| Наблюдения | Конфиг объединяет пользовательские параметры (SFW, шаблоны) и инфраструктурные настройки (пути к БД, device), без разграничения по слоям. 【F:imagen_lab/config.py†L1-L195】【F:imagen_lab/config.py†L196-L292】【F:config.yaml†L1-L60】

## Карта текущих функций → целевые слои

| Модуль / функция | Существующая роль | Целевой слой |
| --- | --- | --- |
| `SceneBuilder.build_scene` | Выбор шаблона, wardrobe, персонажа, формирование payload для Ollama | `scene` (структурирование) + частично `caption` (payload требования) |
| `SceneBuilder.rebuild_from_genes` | Повторная сборка сцены для GA | `scene` |
| `PromptComposer.system_prompt` | Построение системного промпта Ollama | `caption` |
| `PromptComposer.generate_caption` (через `PromptWorkflow`) | Вызов Ollama, enforcement терминов | `caption` |
| `PromptComposer.final_prompt` | Пост-обработка текста под ограничения Imagen | `image` (адаптация под генератор) |
| `DualScorer.score_one` / `score_and_save` | Вычисление микрометрик и агрегация, запись в БД | `scoring` + `db` |
| `save_and_score` | Обработка Imagen-ответа, сохранение файлов, логирование, агрегация метрик | `image` (сохранение артефактов) + `scoring` (агрегаты) + `db` (логирование) |
| `_prepare_services` | Инициализация всех зависимостей | оркестратор, кандидаты на разделение по слоям |
| `PipelineConfig.from_dict` | Загрузка и нормализация настроек | `scene`/`caption`/`image`/`scoring`/`db` (нужна декомпозиция по слоям) |

## Выводы

1. Большинство функций работает на пересечении нескольких слоёв (например, `SceneBuilder` уже готовит JSON для `caption`, а `save_and_score` агрегирует метрики скоринга и пишет в БД).
2. Чтобы достичь желаемого разделения, необходимо изолировать API между слоями (структуры сцены, контракт caption, контракт image response, scoring DTO).
3. Конфигурация инициализирует все слои сразу, что затрудняет частичное переиспользование; следует разбить её на блоки (`scene`, `caption`, `image`, `scoring`, `db`).

