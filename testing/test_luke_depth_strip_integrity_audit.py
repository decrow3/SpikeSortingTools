from testing.luke_depth_strip_integrity_audit import flatten_strings, nested_values


def test_nested_values_collects_dicts_and_lists():
    graph = {
        "class": "outer",
        "children": [{"class": "inner"}, {"other": {"class": "deep"}}],
    }
    assert nested_values(graph, "class") == ["outer", "inner", "deep"]


def test_flatten_strings_flattens_path_lists():
    assert flatten_strings([["a", "b"], "c", 4]) == ["a", "b", "c"]
