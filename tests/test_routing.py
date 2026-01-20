from app.main import infer_category, filter_results

def test_infer_category_billing():
    assert infer_category("How long is the refund window for Pro?") == "billing"

def test_filter_results_by_category():
    results = [
        {"chunk_id": "a", "metadata": {"category": "billing", "applies_to": "Pro, Team"}},
        {"chunk_id": "b", "metadata": {"category": "privacy", "applies_to": "Free, Pro, Team"}}
    ]

    out = filter_results(results, category = "privacy", applies_to = None)
    assert [r["chunk_id"] for r in out] == ["b"]