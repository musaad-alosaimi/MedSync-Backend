"""Offline tests for server/vision.py (a fake client stands in for the API, so no key or network is needed).
Run from the repo root:  server/.venv/bin/python -m unittest server.test_vision -v
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                      # noqa: E402
import cv2                              # noqa: E402
import interaction_checker as ic        # noqa: E402
from server import vision               # noqa: E402

KB = ic.KnowledgeBase()


def fake_client(payload=None, finish="stop", text=None, raises=None, calls=None):
    """Stands in for an OpenAI-compatible client: client.chat.completions.create(...)."""
    def create(**kw):
        if calls is not None:
            calls.append(kw)
        if raises:
            raise raises
        body = text if text is not None else json.dumps(payload)
        return NS(choices=[NS(finish_reason=finish, message=NS(content=body))])
    return NS(chat=NS(completions=NS(create=create)))


def item(**kw):
    base = {"brand": None, "ingredients": [], "strength": None, "form": None, "legibility": "clear", "note": None}
    return base | kw


def jpeg(w=800, h=600):
    img = np.full((h, w, 3), 200, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class VisionTests(unittest.TestCase):
    def test_reported_text_is_resolved_through_the_database(self):
        c = fake_client({"items": [item(brand="Brufen", ingredients=["Ibuprofen"], strength="400 mg"),
                                   item(ingredients=["Warfarin"])]})
        pkgs, seen = vision.read_packages(KB, jpeg(), c)
        self.assertEqual([p.ingredients for p in pkgs], [["ibuprofen"], ["warfarin"]])
        self.assertEqual(pkgs[0].strengths.get("ibuprofen"), "400 mg")
        self.assertEqual(seen[1]["brand"], "Brufen")
        found, _, _ = ic.check_interactions(KB, pkgs)
        self.assertTrue(found and found[0]["risk"] == "Major")

    def test_a_name_the_database_does_not_know_stays_unidentified(self):
        c = fake_client({"items": [item(brand="Zzzqxplorix", ingredients=["Madeupamine"], legibility="partial")]})
        pkgs, seen = vision.read_packages(KB, jpeg(), c)
        self.assertEqual(pkgs[0].status, "unidentified")
        self.assertEqual(pkgs[0].ingredients, [])
        self.assertEqual(seen[1]["brand"], "Zzzqxplorix")          # still reported, so the user can confirm it

    def test_two_identical_boxes_are_a_duplicate(self):
        c = fake_client({"items": [item(brand="Brufen", ingredients=["Ibuprofen"]),
                                   item(brand="Nurofen", ingredients=["Ibuprofen"])]})
        pkgs, _ = vision.read_packages(KB, jpeg(), c)
        self.assertEqual([d["ingredient"] for d in ic.find_duplicates(pkgs)], ["ibuprofen"])

    def test_guessed_items_are_flagged(self):
        c = fake_client({"items": [item(brand="Brufen", legibility="guess")]})
        pkgs, _ = vision.read_packages(KB, jpeg(), c)
        self.assertTrue(any("only guess" in n for n in pkgs[0].notes))

    def test_every_failure_is_a_visionerror_so_the_caller_can_fall_back(self):
        for c in (fake_client(finish="length", payload={"items": []}),
                  fake_client(text="not json"),
                  fake_client(text=""),
                  fake_client(text=json.dumps({"items": "oops"})),
                  fake_client(raises=ConnectionError("down"))):
            with self.assertRaises(vision.VisionError):
                vision.read_packages(KB, jpeg(), c)
        with self.assertRaises(vision.VisionError):
            vision.read_packages(KB, b"not an image", fake_client({"items": []}))

    def test_placeholder_and_empty_items_are_dropped(self):
        c = fake_client({"items": [item(brand="Unknown", legibility="guess"), item(brand="N/A"),
                                   item(brand=None, legibility="guess", note="white box, label cut off"),
                                   item(brand="Brufen", ingredients=["Ibuprofen"])]})
        pkgs, _ = vision.read_packages(KB, jpeg(), c)
        self.assertEqual(len(pkgs), 2)                              # the described box and the Brufen box

    def test_code_fenced_json_is_accepted(self):
        c = fake_client(text="```json\n" + json.dumps({"items": [item(brand="Brufen")]}) + "\n```")
        pkgs, _ = vision.read_packages(KB, jpeg(), c)
        self.assertEqual(pkgs[0].ingredients, ["ibuprofen"])

    def test_a_cut_off_answer_keeps_the_complete_items(self):
        whole = json.dumps({"items": [item(brand="Brufen", ingredients=["Ibuprofen"]), item(brand="Panadol")]})
        cut = whole[: whole.rindex('"brand"') + 12]                  # the second item is cut mid-string
        pkgs, _ = vision.read_packages(KB, jpeg(), fake_client(text=cut))
        self.assertEqual([p.ingredients for p in pkgs], [["ibuprofen"]])

    def test_a_failed_request_is_retried_once(self):
        calls, state = [], {"n": 0}
        ok = fake_client({"items": [item(brand="Brufen")]})
        def flaky(**kw):
            state["n"] += 1
            calls.append(kw)
            if state["n"] == 1:
                raise ConnectionError("blip")
            return ok.chat.completions.create(**kw)
        pkgs, _ = vision.read_packages(KB, jpeg(), NS(chat=NS(completions=NS(create=flaky))))
        self.assertEqual(len(calls), 2)
        self.assertEqual(pkgs[0].ingredients, ["ibuprofen"])

    def test_a_padded_list_is_marked_low_trust(self):
        names = ["Brufen", "Panadol", "Zyrtec", "Claritin", "Mucinex", "Sudafed", "Aleve", "Motrin", "Tums", "Pepcid",
                 "Prilosec"]
        c = fake_client({"items": [item(brand=n) for n in names]})
        _, seen = vision.read_packages(KB, jpeg(), c)
        self.assertTrue(all(v["low_trust"] for v in seen.values()))
        _, seen = vision.read_packages(KB, jpeg(), fake_client({"items": [item(brand="Brufen"), item(brand="Panadol", legibility="partial")]}))
        self.assertEqual([seen[1]["low_trust"], seen[2]["low_trust"]], [False, True])

    def test_item_count_is_capped(self):
        c = fake_client({"items": [item(brand=f"Brand{i}") for i in range(80)]})   # model padding its answer
        pkgs, _ = vision.read_packages(KB, jpeg(), c)
        self.assertEqual(len(pkgs), vision.MAX_ITEMS)

    def test_a_repetition_loop_is_collapsed_and_a_cut_off_answer_is_never_trusted(self):
        loop = [item(brand="Cetirizine", ingredients=["Cetirizine"])] * 16
        pkgs, _ = vision.read_packages(KB, jpeg(), fake_client({"items": loop}))
        self.assertEqual(len(pkgs), vision.MAX_IDENTICAL)           # two identical boxes are plausible, sixteen are a loop
        cut = json.dumps({"items": [item(brand="Brufen", ingredients=["Ibuprofen"])]})[:-2]     # never closed
        _, seen = vision.read_packages(KB, jpeg(), fake_client(text=cut))
        self.assertTrue(seen[1]["low_trust"])

    def test_upload_is_downscaled_and_reencoded(self):
        out = vision.prepare_image(jpeg(4000, 3000))
        h, w = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        self.assertEqual(max(h, w), vision.MAX_SIDE)

    def test_availability_follows_the_environment(self):
        import os
        names = ("MEDSYNC_VLM", "MEDSYNC_VLM_API_KEY", "MEDSYNC_VLM_BASE_URL")
        keep = {k: os.environ.get(k) for k in names}
        try:
            for k in names:
                os.environ.pop(k, None)
            self.assertFalse(vision.available())
            os.environ["MEDSYNC_VLM_API_KEY"] = "x"
            self.assertFalse(vision.available())                   # a key alone is not enough: needs the endpoint
            os.environ["MEDSYNC_VLM_BASE_URL"] = "https://example.invalid/v1"
            self.assertTrue(vision.available())
            os.environ["MEDSYNC_VLM"] = "off"
            self.assertFalse(vision.available())
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
