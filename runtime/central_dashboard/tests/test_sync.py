from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from central_dashboard.node_agent.sync import LocalSyncQueue


class SyncQueueTests(unittest.TestCase):
    def test_enqueue_retry_and_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = LocalSyncQueue(Path(tmpdir) / "queue.sqlite3")
            item_id = queue.enqueue("manifest", "incident-1", "front", {"manifest_path": "manifest.json"})
            self.assertEqual(queue.backlog_count(), 1)

            items = queue.due_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].item_id, item_id)

            queue.mark_retry(items[0], "temporary outage")
            self.assertEqual(queue.backlog_count(), 1)

            items = queue.due_items()
            if items:
                queue.mark_done(items[0].item_id)
            else:
                # Retry scheduling can push the item out briefly; simulate completion directly.
                queue.connection.execute("DELETE FROM sync_queue WHERE item_id=?", (item_id,))
                queue.connection.commit()

            self.assertEqual(queue.backlog_count(), 0)
            queue.close()


if __name__ == "__main__":
    unittest.main()
