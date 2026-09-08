# KrishiDisha FastAPI service (optional)

A standalone, Swagger-documented HTTP API over the same engine as the website.
Use it for mobile apps, IVR / SMS / WhatsApp gateways, or headless deployments.
The website itself already exposes the same endpoints under `/api/v1/` (see
`krishidisha/blueprints/api.py`), so you only need this module if you want the
API without the Flask UI, or FastAPI's interactive docs.

```bash
pip install -r requirements.txt            # project root requirements include fastapi + uvicorn
uvicorn api.main:app --reload --port 8000  # docs at http://localhost:8000/docs
pytest api/test_api.py                     # offline smoke tests
```

Configuration comes from the same `.env` as the website (`LLM_PROVIDER`,
`DISEASE_MODEL_BACKEND`, `DATABASE_URL`, ...).

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/health` | | provider, model status, disease model |
| GET | `/reference` | | valid soil / crop / state / season names |
| POST | `/crop/recommend` | N, P, K, temperature, humidity, ph, rainfall | crop, confidence, alternatives, economics, guide |
| POST | `/fertilizer/recommend` | temperature, humidity, moisture, soil_type, crop_type, N, P, K, area?, unit? | fertilizer, NPK, note, dose calculator |
| POST | `/fertilizer/calculator` | crop, area, unit?, soil_n?, soil_p?, soil_k? | kg and bags of Urea / DAP / MOP, schedule |
| POST | `/yield/predict` | crop, crop_year, season, state, area, production, annual_rainfall, fertilizer, pesticide | tonnes/ha, production, tips |
| POST | `/disease/detect` | multipart `image` | predictions, disease info, products |
| POST | `/chat` | message, language?, history?, farmer_context?, plain? | reply (markdown), tools used, sources |
| POST | `/chat/image` | multipart `image`, message?, language? | reply grounded in the leaf analysis |
| GET | `/weather` | place, days? | forecast + advisories |
| GET | `/mandi` | commodity, state?, district?, limit? | live Agmarknet prices or MSP fallback |
| GET | `/msp` | commodity? | MSP reference table |
| GET | `/schemes` | q? | government schemes |
| GET | `/crop-guide/{crop}` | | cultivation guide + calendar |
| GET | `/crop-calendar` | crop?, season?, month? | sowing / harvest windows |
| GET | `/diseases` | | PlantVillage disease catalogue |
| GET | `/knowledge/search` | q, k? | ranked knowledge-base passages |
| GET | `/products` | q?, category?, disease?, crop?, limit? | marketplace products |

Example:

```bash
curl -X POST http://localhost:8000/crop/recommend -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":21,"humidity":82,"ph":6.5,"rainfall":203}'
```
