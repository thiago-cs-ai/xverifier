"""A selected cutoff that clears no rows still prints."""

import numpy as np

from xverifier.cli import format_table
from xverifier.decide import count_decisions


class TestZeroDecisionsStillPrint:
    def test_selected_cutoff_with_no_cleared_rows(self):
        # GIVEN a live cutoff of 0.99 and five scores of 0
        cutoffs = [
            {"error_level": 0.10, "review_all": False, "cutoff": 0.99},
            {"error_level": 0.05, "review_all": True, "cutoff": None},
        ]
        points = count_decisions(cutoffs, np.zeros(5), np.zeros(5, dtype=int))
        # WHEN the table is rendered
        table = format_table(points)
        # THEN the empty prefix shows the cutoff and 0/5, and review-all is unchanged
        assert points[0]["decided"] == 0
        assert points[0]["review_all"] is False
        assert points[0]["error_rate"] is None
        assert isinstance(table, str)
        assert table.splitlines() == [
            "error level   cutoff     decided    errors   error rate",
            "0.10          0.990000   0/5",
            "0.05          review all    0/5",
        ]
