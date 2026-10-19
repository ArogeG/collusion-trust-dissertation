from src.sim.parse import parse_position

OPTIONS = ["Vendor A", "Vendor B", "Vendor C", "Vendor D"]


def test_parses_marker_line():
    text = 'Scores favour B.\nPOSITION: {"position": "Vendor B", "confidence": 0.8, "cites": ["a2"]}'
    assert parse_position(text, OPTIONS) == {
        "position": "Vendor B", "confidence": 0.8, "cites": ["a2"], "parsed": True}


def test_uses_last_marker_and_normalises_case():
    text = ('Earlier: POSITION: {"position": "Vendor A", "confidence": 0.5, "cites": []}\n'
            'Revised.\nPOSITION: {"position": "vendor c", "confidence": 0.9, "cites": "a1"}')
    out = parse_position(text, OPTIONS)
    assert out["position"] == "Vendor C" and out["cites"] == ["a1"] and out["parsed"]


def test_falls_back_to_any_json_and_clamps_confidence():
    out = parse_position('Final answer {"position": "Vendor D", "confidence": 3}', OPTIONS)
    assert out["position"] == "Vendor D" and out["confidence"] == 1.0


def test_braces_inside_strings_do_not_break_parsing():
    text = 'POSITION: {"position": "Vendor B", "confidence": 0.7, "cites": ["a{1}"]}'
    assert parse_position(text, OPTIONS)["cites"] == ["a{1}"]


def test_missing_or_invalid_block_is_flagged():
    assert parse_position("I think Vendor B.", OPTIONS)["parsed"] is False
    bad = parse_position('POSITION: {"position": "Vendor Z", "confidence": 0.5}', OPTIONS)
    assert bad["position"] == "Vendor Z" and bad["parsed"] is False
    assert parse_position('POSITION: {"position": "Vendor B", "confidence": "high"}',
                          OPTIONS)["confidence"] is None
