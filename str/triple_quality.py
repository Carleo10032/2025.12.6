from typing import List, Dict, Any


class TripleQualityController:
    def __init__(self, enable_fuzzy_dedup: bool = True):
        self.enable_fuzzy_dedup = enable_fuzzy_dedup

    def deduplicate_triples(self, triples: List[Dict[str, Any]], use_global: bool = True) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for t in triples or []:
            s = (t.get("subject") or "").strip().lower()
            p = (t.get("predicate") or "").strip().lower()
            o = (t.get("object") or "").strip().lower()
            key = (s, p, o)
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def filter_by_quality(self, triples: List[Dict[str, Any]], min_score: float = 0.5) -> List[Dict[str, Any]]:
        def score(t: Dict[str, Any]) -> float:
            s = len((t.get("subject") or "").strip())
            p = len((t.get("predicate") or "").strip())
            o = len((t.get("object") or "").strip())
            if s < 2 or p < 2 or o < 2:
                return 0.0
            return min(1.0, (s + p + o) / 30.0)
        return [t for t in triples or [] if score(t) >= min_score]

