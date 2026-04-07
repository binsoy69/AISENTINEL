Runtime-created files for the standalone central dashboard stack live here.

Suggested layout at runtime:

- `central_service/central.sqlite3`
- `central_service/evidence/`
- `node_front/queue.sqlite3`
- `node_front/evidence/`
- `node_mid/queue.sqlite3`
- `node_mid/evidence/`

These files are intentionally isolated from the legacy front-node runtime.
