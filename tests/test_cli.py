from regexsmith.cli import main


def test_draw_prints_a_diagram(capsys):
    assert main(["draw", r"\d+-[a-z]?"]) == 0
    assert "●" in capsys.readouterr().out


def test_draw_writes_svg(tmp_path):
    out = tmp_path / "d.svg"
    assert main(["draw", "a|b", "--svg", str(out)]) == 0
    assert out.read_text().startswith("<svg")
