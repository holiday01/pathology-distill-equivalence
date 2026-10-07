"""The teacher panel the manuscript reports on.

Hibou-B and Hibou-L are excluded. Their distillation runs were trained under
ImageNet channel statistics, which the hibou encoders do not use, so the
teacher features those students were fit against are not the features the
encoders produce. The six affected runs are therefore not comparable with the
rest of the atlas and are not reported.

Every script that produces a manuscript figure, table or macro filters through
here, so the panel cannot drift between artefacts.
"""

EXCLUDED_TEACHERS = {"hibou-b", "hibou-l", "hibou_base", "hibou_large", "Hibou-B", "Hibou-L"}


def keep_teacher(name):
    """True if this teacher belongs to the reported panel."""
    return str(name).strip().lower().replace("_", "-") not in {
        t.lower().replace("_", "-") for t in EXCLUDED_TEACHERS}


def filter_rows(rows, key="teacher"):
    """Drop excluded teachers from a list of result dicts."""
    return [r for r in rows if keep_teacher(r.get(key, ""))]
