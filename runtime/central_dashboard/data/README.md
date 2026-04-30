Runtime-created files for the standalone central dashboard stack live here.

Suggested layout at runtime:

- `central_service/central.sqlite3`
- `central_service/evidence/`
- `node_front/evidence/`
- `node_front/sound/ky037_ads1015_config.json`
- `node_mid/evidence/`
- `node_mid/sound/ky037_ads1015_config.json`

Node evidence upload uses an in-memory bounded queue only; no SQLite upload
queue is created or resumed.
