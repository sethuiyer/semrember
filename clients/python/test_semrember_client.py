import io
import unittest
import urllib.error
from unittest.mock import patch

from semrember_client import Semrember, SemremberError


class ClientErrorsTest(unittest.TestCase):
    def test_get_returns_none_only_for_http_404(self):
        for status in (404, 409, 500):
            error = urllib.error.HTTPError(
                "http://localhost", status, "failure", {},
                io.BytesIO(b'{"error":"not found in semantic state"}'),
            )
            with self.subTest(status=status), patch("urllib.request.urlopen", side_effect=error):
                if status == 404:
                    self.assertIsNone(Semrember().get("disk"))
                else:
                    with self.assertRaises(SemremberError) as caught:
                        Semrember().get("disk")
                    self.assertEqual(caught.exception.status, status)

    def test_wraps_connection_and_timeout_errors(self):
        for error in (urllib.error.URLError("offline"), TimeoutError("timeout")):
            with self.subTest(error=error), patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(SemremberError):
                    Semrember().ping()

    def test_rejects_non_object_response(self):
        response = io.BytesIO(b"[]")
        response.status = 200
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(SemremberError, "JSON object"):
                Semrember().ping()

    def test_encodes_operation_like_keys_as_one_segment(self):
        self.assertEqual(Semrember._key_path("a/exists+%"), "/v1/kv/a%2Fexists%2B%25")

    def test_influence_and_compact(self):
        influence = io.BytesIO(b'{"memories":[{"key":"disk","beta":0.5}]}')
        influence.status = 200
        with patch("urllib.request.urlopen", return_value=influence):
            self.assertEqual(Semrember().influence(["disk"])[0]["beta"], 0.5)

        compact = io.BytesIO(b'{"ok":true,"kept":3,"pruned":2,"verified":true}')
        compact.status = 200
        with patch("urllib.request.urlopen", return_value=compact):
            result = Semrember().compact(["disk"], coverage_floor=1)
            self.assertTrue(result["verified"])
            self.assertEqual(result["pruned"], 2)


if __name__ == "__main__":
    unittest.main()
