# KrishiDisha 🌾

AI decision support for Indian farmers: crop and fertilizer recommendations, leaf-photo disease detection, yield prediction, weather advisories, live mandi prices, government scheme lookup, a farm-input marketplace and a multilingual agricultural assistant (LLM with tool use, or a fully offline rules engine).

Built by Abhishek Chourasia, Goutam Mandloi and Shivpratap Singh Panwar.

## Quick start

With [uv](https://docs.astral.sh/uv/) (recommended, one command creates the environment from `pyproject.toml` + `uv.lock`):

```bash
uv sync --extra api               # creates .venv with CPU-only PyTorch; drop --extra api if you don't need FastAPI
copy .env.example .env            # optional: add API keys, switch DB, etc.
uv run app.py                     # http://localhost:5000
uv run pytest                     # tests
```

With plain pip:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python app.py
```

* Default admin: `admin` / `admin123` (change with `DEFAULT_ADMIN_*` in `.env` or `flask --app app create-admin NAME PASS`).
* The database is SQLite in `instance/` by default; set `DATABASE_URL` for MySQL.
* The tabular models ship in `models/`; if missing they are trained on first use (seconds). Run `python -m ml.train_all` for a full retrain with evaluation reports.
* Farmers must be verified by an admin before login (as in the project report). Set `AUTO_VERIFY_FARMERS=true` for demos.
* Warm everything up once (models, disease network, assistant): `flask --app app warmup`.

## Features

| Area | What it does | Where |
|---|---|---|
| Crop recommendation | 22 crops from N, P, K, temperature, humidity, pH, rainfall; top-3 with probabilities, indicative economics, cultivation guide, PDF report | `/crop_recommendation` |
| Fertilizer recommendation | Urea / DAP / complex grades from soil and crop type, plus a kg-and-bags dose calculator with split schedule, PDF report, matching products | `/fertilizer_recommendation`, `/tools/fertilizer-calculator` |
| Disease detection | Upload a leaf photo (38 PlantVillage classes), description, prevention steps, matching marketplace products, PDF report | `/crop_detection` |
| Yield prediction | Tonnes/ha for 55 crops × 30 states × 6 seasons from historical data | `/crop_yield` |
| Weather | 7-day forecast, soil moisture, and rule-based agro-advisories (rain, heat, frost, spraying windows) via Open-Meteo | `/weather` |
| Mandi prices | Live Agmarknet prices from data.gov.in with MSP fallback | `/mandi` |
| Schemes & calendar | 16 central schemes with eligibility and how to apply; sowing/harvest calendar; 32 crop guides; pest management | `/schemes`, `/crop-calendar`, `/crop-guide/<crop>` |
| Marketplace | 60 seeded products (fertilizers, fungicides, insecticides, seeds, organics, tools), cart, checkout (COD/UPI), order tracking, stock control | `/marketplace` |
| Assistant | Chat with tool use over every service above, leaf photo analysis, Hindi/Hinglish/regional languages, conversation history | `/chat` and the floating widget |
| Admin | Farmer verification and CRUD, orders, catalogue, activity audit, chat logs | `/admin_dashboard` |
| JSON API | Everything above for apps/SMS/WhatsApp gateways | `/api/v1/...` (see `/about`) |

## The assistant

`LLM_PROVIDER` selects the brain; all providers share the same tools (`krishidisha/services/tools.py`) and the same TF-IDF retrieval over the knowledge base, so answers are grounded in the project's own data and models.

| Provider | Setup | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude with native tool use and vision (default when a key is present) |
| `openai` | `OPENAI_API_KEY` or `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, local Ollama / LM Studio |
| `rules` | nothing | Offline intent engine that still runs the ML models, weather, prices, calculator, schemes and KB search. Automatic fallback if an API call fails. |

## Project layout

```
app.py                      WSGI entry point (gunicorn app:app)
krishidisha/                Flask package
  __init__.py               app factory, services, CLI (seed, create-admin, warmup)
  config.py                 environment-driven configuration (.env)
  models.py                 Farmer, Admin, FarmerActivity, Product, CartItem, Address, Order, ChatSession ...
  blueprints/               main, auth, farmer, marketplace, admin, chat, api
  services/                 ml, disease, knowledge, tools, llm, fallback_bot, weather, market, reports
ml/                         reproducible training: train_tabular, train_disease, train_all
data/                       CSV datasets + knowledge/*.json (schemes, products, crop guides, calendar, pests, MSP)
models/                     trained artefacts, metrics_tabular.json, evaluation plots in models/reports
templates/, static/         Bootstrap 5 UI
tests/                      pytest suite (offline; in-memory SQLite)
api/                        optional standalone FastAPI service (legacy)
```

## Training

```bash
python -m ml.train_tabular --data-dir data --output models        # compares RF / GB / XGBoost / SVM / kNN / NB ...
python -m ml.train_disease --data-dir <PlantVillage root> --epochs 6 --arch mobilenet_v3_large
python -m ml.train_all                                            # both (image stage skipped if dataset absent)
```

Latest tabular results (`models/metrics_tabular.json`): crop recommendation 99.5 % test accuracy (GaussianNB, RF 99.3 %), fertilizer 100 % (XGBoost), yield R² 0.94 to 0.99 (best CV: XGBoost). Plots are in `models/reports/`.

For disease detection without training: `DISEASE_MODEL_BACKEND=hf` downloads a pretrained PlantVillage MobileNet from the Hugging Face Hub on first use. The original `plant_disease_model_1_latest.pt` from `CNN.py` is also supported (`legacy`).

## Tests

```bash
pytest
```

The suite runs fully offline (rules assistant, stub disease model, in-memory database).

## Deployment

`Procfile` runs `gunicorn app:app`. Set `SECRET_KEY`, `DATABASE_URL`, `AUTO_VERIFY_FARMERS` and any API keys as environment variables. Uploads go to `static/uploads/` (`UPLOAD_DIR`).

## Disclaimer

Recommendations are estimates from models trained on public datasets. Validate with your local Krishi Vigyan Kendra or an agronomist before large investments. Kisan Call Centre: 1800-180-1551.
