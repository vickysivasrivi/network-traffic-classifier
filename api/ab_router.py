"""Champion-challenger A/B routing for the Network Threat Detection API.

Supports two operating modes:

* **Shadow mode** (default): Both models score every request; only the champion
  result is returned to the caller. The challenger result is discarded but its
  Prometheus counter is incremented for offline comparison.

* **Traffic-split mode**: A configurable fraction (``traffic_split``) of live
  requests is routed to the challenger and its result returned to the caller.
  Remaining requests go to the champion.

If no challenger path is configured, or the file does not exist, the router
falls back transparently to the champion only.
"""

import os
import random

from api.monitoring import AB_CHAMPION_COUNTER, AB_CHALLENGER_COUNTER
from api.predictor import Predictor
from api.schemas import PredictionResponse


class ABRouter:
    """Route inference requests between a champion and an optional challenger model.

    Args:
        champion:               Loaded :class:`~api.predictor.Predictor` instance
                                used as the primary (champion) model.
        challenger_model_path:  Absolute or relative path to a directory that
                                contains the challenger ``best_model.joblib``.
                                If ``None`` or the path does not exist the router
                                operates in champion-only mode.
        traffic_split:          Fraction of requests routed to the challenger in
                                traffic-split mode.  ``0.0`` means champion only;
                                ``1.0`` means challenger only.  Ignored in shadow
                                mode.
        shadow_mode:            When ``True`` (default) the champion result is
                                always returned and the challenger runs silently.
    """

    def __init__(
        self,
        champion: Predictor,
        challenger_model_path: str | None = None,
        traffic_split: float = 0.0,
        shadow_mode: bool = True,
    ) -> None:
        self.champion = champion
        self.traffic_split = float(traffic_split)
        self.shadow_mode = bool(shadow_mode)
        self.challenger: Predictor | None = None

        if challenger_model_path and os.path.exists(challenger_model_path):
            self.challenger = Predictor(models_dir=challenger_model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, features: list[float]) -> tuple[PredictionResponse, str]:
        """Score a feature vector and return (result, model_name).

        Args:
            features: 70 raw feature values in ``feature_names.txt`` order.

        Returns:
            Tuple of ``(PredictionResponse, served_by)`` where ``served_by`` is
            ``"champion"`` or ``"challenger"``.
        """
        if self.challenger is None:
            # No challenger loaded — always serve champion.
            AB_CHAMPION_COUNTER.inc()
            return self.champion.predict(features), "champion"

        if self.shadow_mode:
            # Shadow mode: run both, return champion result only.
            result = self.champion.predict(features)
            try:
                self.challenger.predict(features)  # result logged via counter, then discarded
            except Exception:
                pass
            AB_CHAMPION_COUNTER.inc()
            AB_CHALLENGER_COUNTER.inc()
            return result, "champion"

        # Traffic-split mode: probabilistically route to challenger.
        if random.random() < self.traffic_split:
            AB_CHALLENGER_COUNTER.inc()
            return self.challenger.predict(features), "challenger"

        AB_CHAMPION_COUNTER.inc()
        return self.champion.predict(features), "champion"

    @property
    def challenger_loaded(self) -> bool:
        """``True`` if a challenger model has been successfully loaded."""
        return self.challenger is not None
