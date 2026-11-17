"""Phenotypic clustering of objects on their standardised features.

Clusters are learned once on a sample of objects from the whole plate, then every object
is assigned: k-means by its nearest centre; HDBSCAN and Leiden (which have no exact rule for
new points) by the majority of its 5 nearest sampled neighbours, an approximation recorded
in the step's summary. Cluster 1 is the largest; -1 is HDBSCAN noise.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.classifier import feature_matrix, robust_stats
from hcs_studio.analysis.table_steps import (
    TableModule,
    TableParams,
    WellContext,
    count_fractions,
    feature_columns,
)
from hcs_studio.processing.base import MODULES, ModuleError

#: Standardised values are clipped to ±CLIP so a few extreme objects cannot dominate.
CLIP = 10.0


class ClusterParams(TableParams):
    name: str = Field(
        default="pheno", pattern=r"^[A-Za-z0-9_]+$", description="Objects get cluster_<name>."
    )
    features: list[str] | None = Field(
        default=None, description="Features to cluster on; empty = all measured features."
    )
    method: Literal["kmeans", "hdbscan", "leiden"] = Field(
        default="kmeans", description="Leiden needs the optional leidenalg and igraph packages."
    )
    n_clusters: int = Field(default=5, ge=2, le=100, description="k-means: number of clusters.")
    min_cluster_size: int = Field(default=50, ge=2, description="HDBSCAN: smallest cluster.")
    resolution: float = Field(default=1.0, gt=0, description="Leiden: higher = more clusters.")
    n_neighbors: int = Field(default=15, ge=2, le=200, description="Leiden: k-NN graph size.")
    pca_components: int = Field(
        default=0, ge=0, le=200, description="Reduce to this many principal components; 0 = off."
    )
    max_fit_objects: int = Field(
        default=50_000, ge=100, description="Objects sampled across the plate to learn from."
    )
    seed: int = Field(default=0, description="Random seed (sampling, k-means, Leiden).")


def _relabel_by_size(labels: NDArray[Any]) -> tuple[NDArray[np.int64], dict[int, int]]:
    """Clusters renumbered 1..k from largest to smallest; noise stays -1."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    mapping = {int(old): new for new, old in enumerate(order, start=1)}
    out = np.array([mapping.get(int(v), -1) for v in labels], dtype=np.int64)
    return out, mapping


def _leiden(x: NDArray[Any], n_neighbors: int, resolution: float, seed: int) -> NDArray[Any]:
    try:
        import igraph
        import leidenalg
    except ImportError:
        raise ModuleError(
            "Leiden clustering needs the optional packages leidenalg and igraph "
            "(pip install leidenalg igraph); or choose k-means or HDBSCAN"
        ) from None
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(x, n_neighbors=min(n_neighbors, len(x) - 1), include_self=False)
    graph = graph.maximum(graph.T).tocoo()
    edges = [(int(i), int(j)) for i, j in zip(graph.row, graph.col, strict=True) if i < j]
    g = igraph.Graph(n=len(x), edges=edges)
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed
    )
    return np.asarray(partition.membership)


@MODULES.register("phenotypic_clusters")
class PhenotypicClusters(TableModule):
    """Unsupervised phenotypes: k-means, HDBSCAN or Leiden on robust-standardised per-object
    features; per-well fraction of objects in each cluster."""

    name = "phenotypic_clusters"
    title = "Phenotypic clusters"
    Params = ClusterParams
    needs_fit = True

    def _matrix(self, frame: pd.DataFrame, state: dict[str, Any]) -> NDArray[np.float64]:
        x = feature_matrix(frame, state["features"], state["stats"], state["fill"])
        x = np.clip(x, -CLIP, CLIP)
        if state["pca"] is not None:
            x = state["pca"].transform(x)
        return x

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        params: ClusterParams = self.params
        names = feature_columns(sample, params.features)
        names = [
            n for n in names if pd.to_numeric(sample[n], errors="coerce").nunique(dropna=True) > 1
        ]
        if not names:
            raise ModuleError("no features vary between objects")
        minimum = params.n_clusters if params.method == "kmeans" else params.min_cluster_size
        if len(sample) < max(10, minimum):
            raise ModuleError(f"only {len(sample)} objects to cluster")
        stats = robust_stats(sample, names)
        state: dict[str, Any] = {
            "features": names,
            "stats": stats,
            "fill": dict.fromkeys(names, 0.0),
            "pca": None,
        }
        x = np.clip(feature_matrix(sample, names, stats, state["fill"]), -CLIP, CLIP)
        if params.pca_components and params.pca_components < len(names):
            from sklearn.decomposition import PCA

            state["pca"] = PCA(n_components=params.pca_components, random_state=params.seed).fit(x)
            x = state["pca"].transform(x)
        if params.method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=params.n_clusters, n_init=10, random_state=params.seed)
            labels = model.fit_predict(x)
            state["kmeans"] = model
        elif params.method == "hdbscan":
            from sklearn.cluster import HDBSCAN

            labels = HDBSCAN(min_cluster_size=params.min_cluster_size, copy=True).fit_predict(x)
        else:
            labels = _leiden(x, params.n_neighbors, params.resolution, params.seed)
        labels, mapping = _relabel_by_size(np.asarray(labels))
        state["mapping"] = mapping
        if params.method != "kmeans":
            from sklearn.neighbors import KNeighborsClassifier

            state["assign"] = KNeighborsClassifier(n_neighbors=min(5, len(x))).fit(x, labels)
        ids, counts = np.unique(labels, return_counts=True)
        state["sizes"] = {int(k): int(v) for k, v in zip(ids, counts, strict=True)}
        state["profiles"] = self._profiles(sample, names, stats, labels)
        state["fitted_objects"] = len(sample)
        return state

    @staticmethod
    def _profiles(
        sample: pd.DataFrame, names: list[str], stats: Any, labels: NDArray[Any]
    ) -> dict[int, list[tuple[str, float]]]:
        """Per cluster, the features whose median Z-score differs most from the plate."""
        z = np.clip(feature_matrix(sample, names, stats, dict.fromkeys(names, 0.0)), -CLIP, CLIP)
        profiles = {}
        for cluster in sorted(set(labels.tolist())):
            if cluster < 0:
                continue
            median = np.median(z[labels == cluster], axis=0)
            top = np.argsort(-np.abs(median))[:5]
            profiles[int(cluster)] = [(names[i], round(float(median[i]), 2)) for i in top]
        return profiles

    def assign(self, frame: pd.DataFrame, state: dict[str, Any]) -> NDArray[np.int64]:
        x = self._matrix(frame, state)
        if not len(x):
            return np.zeros(0, dtype=np.int64)
        if "kmeans" in state:
            raw = state["kmeans"].predict(x)
            return np.array([state["mapping"].get(int(v), -1) for v in raw], dtype=np.int64)
        return np.asarray(state["assign"].predict(x), dtype=np.int64)

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        column = pd.Series(pd.NA, index=objects.index, dtype="Int64")
        rows = self.selected(objects)
        if rows.any():
            column[rows] = self.assign(objects[rows], state)
        return pd.DataFrame({f"cluster_{self.params.name}": column})

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        labels = sorted(state["sizes"])
        prefix = f"cluster_{self.params.name}"
        values = objects[prefix].astype("Int64")
        names = {k: ("noise" if k < 0 else str(k)) for k in labels}
        return count_fractions(prefix, values.map(names), [names[k] for k in labels]), {}

    def describe_state(self, state: Any) -> dict[str, Any]:
        return {
            "features": state["features"],
            "fitted_objects": state["fitted_objects"],
            "cluster_sizes_in_sample": state["sizes"],
            "top_features_median_z": state["profiles"],
            "assignment": "nearest centre"
            if "kmeans" in state
            else "majority of 5 nearest sampled objects (approximate)",
        }
