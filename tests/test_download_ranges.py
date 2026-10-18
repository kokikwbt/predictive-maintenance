import importlib
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

module = importlib.import_module("pdmdata.io.download")


class RangeDownloadTest(unittest.TestCase):
    def test_ranges_resume_truncation_and_publish_exact_content(self):
        content = bytes(range(251)) * 100
        requested = []
        lock = threading.Lock()
        truncated = [False]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"fixed"')
                self.end_headers()

            def do_GET(self):
                start, end = map(int, self.headers["Range"][6:].split("-"))
                with lock:
                    requested.append((start, end))
                    short = start == 0 and not truncated[0]
                    if short:
                        truncated[0] = True
                self.send_response(206)
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, len(content)))
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                body = content[start:end + 1]
                self.wfile.write(body[:len(body)//2] if short else body)
                self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary, \
                 patch.object(module, "MIN_RANGE_SIZE", 1), \
                 patch.object(module, "RANGE_CHUNK_SIZE", 2048), \
                 patch.dict(os.environ, {"PDMDATA_DOWNLOAD_WORKERS": "4"}):
                target = Path(temporary) / "download.csv"
                module._download_file("http://127.0.0.1:{}/file".format(server.server_port), target)
                self.assertEqual(target.read_bytes(), content)
                self.assertIn((1024, 2047), requested)
                self.assertFalse(target.with_suffix(".csv.part").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_range_failure_preserves_existing_target(self):
        class Response:
            status = 200
            headers = {"Content-Length": "100", "Accept-Ranges": "bytes"}
            def geturl(self):
                return "http://example.org/data"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(module, "MIN_RANGE_SIZE", 1), \
             patch.object(module, "urlopen", return_value=Response()), \
             patch.object(module.time, "sleep"), \
             patch.dict(os.environ, {"PDMDATA_DOWNLOAD_WORKERS": "2"}):
            target = Path(temporary) / "existing.csv"
            target.write_bytes(b"preserve")
            with self.assertRaisesRegex(OSError, "byte range"):
                module._download_file("http://example.org/data", target)
            self.assertEqual(target.read_bytes(), b"preserve")
            self.assertFalse(target.with_suffix(".csv.part").exists())

    def test_worker_count_is_bounded(self):
        with patch.dict(os.environ, {"PDMDATA_DOWNLOAD_WORKERS": "9"}):
            with self.assertRaisesRegex(ValueError, "between 1 and 8"):
                module._download_file("http://example.org/data", Path("unused"))


if __name__ == "__main__":
    unittest.main()
