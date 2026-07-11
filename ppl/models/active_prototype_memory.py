"""Dynamic active-conformer prototype memory for MIL models."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicActivePrototypeBank(nn.Module):
    """Per-epoch clustered memory of active conformer prototypes.

    Prototypes are buffers rather than trainable parameters. The bank is rebuilt
    once per epoch from that epoch's accumulated active candidates via
    ``rebuild_from_candidates`` — a deterministic, order-invariant, per-series
    clustering (see that method). It is rebuilt from training batches only, so
    validation and test data cannot leak label information into the memory.
    """

    def __init__(
        self,
        dim: int = 256,
        max_prototypes: int = 64,
        min_count_to_keep: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.max_prototypes = int(max_prototypes)
        self.min_count_to_keep = int(min_count_to_keep)
        self.eps = float(eps)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.max_prototypes <= 0:
            raise ValueError(
                f"max_prototypes must be positive, got {max_prototypes}"
            )

        self.register_buffer(
            "prototypes",
            torch.zeros(self.max_prototypes, self.dim),
        )
        self.register_buffer("counts", torch.zeros(self.max_prototypes))
        self.register_buffer(
            "active_mask",
            torch.zeros(self.max_prototypes, dtype=torch.bool),
        )
        self.register_buffer(
            "prototype_series",
            torch.full((self.max_prototypes,), -1, dtype=torch.long),
        )
        self._series_to_id: Dict[str, int] = {}
        self._id_to_series: Dict[int, str] = {}
        self._series_seen_counts: Dict[int, float] = {}

    def get_extra_state(self) -> dict:
        return {
            "series_to_id": dict(self._series_to_id),
            "id_to_series": dict(self._id_to_series),
            "series_seen_counts": dict(self._series_seen_counts),
        }

    def set_extra_state(self, state: dict) -> None:
        state = state or {}
        self._series_to_id = {
            str(key): int(value)
            for key, value in state.get("series_to_id", {}).items()
        }
        self._id_to_series = {
            int(key): str(value)
            for key, value in state.get("id_to_series", {}).items()
        }
        self._series_seen_counts = {
            int(key): float(value)
            for key, value in state.get("series_seen_counts", {}).items()
        }

    def _series_id(self, series_label: object) -> int:
        label = str(series_label)
        if label not in self._series_to_id:
            next_id = len(self._series_to_id)
            self._series_to_id[label] = next_id
            self._id_to_series[next_id] = label
        return self._series_to_id[label]

    def series_ids_for_labels(
        self,
        series_labels: Sequence[object],
        *,
        create: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Resolve series labels to internal IDs.

        Unknown labels become -1 unless ``create=True``. Query-time code should
        use ``create=False`` so validation/test labels do not mutate the bank.
        """
        ids: list[int] = []
        for label in series_labels:
            key = str(label)
            if key in self._series_to_id:
                ids.append(self._series_to_id[key])
            elif create:
                ids.append(self._series_id(key))
            else:
                ids.append(-1)
        return torch.as_tensor(
            ids,
            device=device or self.prototypes.device,
            dtype=torch.long,
        )

    def _series_ids(
        self,
        series_labels: Optional[Sequence[object]],
        expected_len: int,
    ) -> Optional[torch.Tensor]:
        if series_labels is None:
            return None
        if len(series_labels) != expected_len:
            raise ValueError(
                "candidate_series length must match candidates: "
                f"{len(series_labels)} != {expected_len}"
            )
        return self.series_ids_for_labels(
            series_labels,
            create=True,
            device=self.prototypes.device,
        )

    @torch.no_grad()
    def rebuild_from_candidates(
        self,
        candidates: torch.Tensor,
        candidate_series: Optional[Sequence[object]] = None,
        max_per_series: Optional[int] = None,
    ) -> None:
        """Rebuild the whole bank from an epoch's active-conformer candidates.

        Order-invariant by construction: candidates are grouped by congeneric
        series, each series is clustered deterministically into at most
        ``max_per_series`` clusters (the input is canonicalised first), and every
        cluster with at least ``min_count_to_keep`` members becomes one
        count-weighted, L2-normalised prototype. Independent of candidate/batch
        order, so the bank is reproducible for a fixed candidate set regardless of
        shuffling — and prototypes are never mixed across series.

        Parameters
        ----------
        candidates
            Tensor with shape [M, D] (all of this epoch's active candidates).
        candidate_series
            Optional congeneric-series label for each candidate.
        max_per_series
            Cap on prototypes per series (defaults to ``max_prototypes``).
        """
        if candidates is None or candidates.numel() == 0:
            return
        if candidates.dim() != 2 or candidates.size(-1) != self.dim:
            raise ValueError(
                "rebuild_from_candidates expects candidates with shape "
                f"[M, {self.dim}], got {tuple(candidates.shape)}"
            )

        device, dtype = self.prototypes.device, self.prototypes.dtype
        cand = F.normalize(
            candidates.detach().to(device=device, dtype=dtype), p=2, dim=-1
        )
        # Drop non-finite candidates: a diverged/overflowed embedding must neither
        # crash the clustering nor be stored as a NaN prototype that would poison
        # every subsequent query.
        finite = torch.isfinite(cand).all(dim=1)
        if not bool(finite.all()):
            cand = cand[finite]
            if candidate_series is not None:
                candidate_series = [
                    s for s, keep in zip(candidate_series, finite.tolist()) if keep
                ]
        if cand.size(0) == 0:
            return

        series_ids = self._series_ids(candidate_series, cand.size(0))
        cap = (
            self.max_prototypes
            if not max_per_series or int(max_per_series) <= 0
            else int(max_per_series)
        )

        if series_ids is None:
            groups = {-1: torch.arange(cand.size(0), device=device)}
        else:
            positions: Dict[int, list] = {}
            for pos, sid in enumerate(series_ids.tolist()):
                positions.setdefault(int(sid), []).append(pos)
                self._series_seen_counts[int(sid)] = (
                    self._series_seen_counts.get(int(sid), 0.0) + 1.0
                )
            groups = {
                sid: torch.as_tensor(pos, device=device, dtype=torch.long)
                for sid, pos in positions.items()
            }

        # Build the new prototype set first, WITHOUT touching the live bank.
        # Series are ordered by stable label (not encounter-order id) so the slot
        # layout is reproducible across data orderings.
        new: list = []  # (series_id, centroid, count)
        for sid in sorted(groups, key=lambda s: self._id_to_series.get(int(s), str(s))):
            if len(new) >= self.max_prototypes:
                break
            centroids, sizes = self._cluster_centroids(cand[groups[sid]], cap)
            for centroid, size in zip(centroids, sizes):
                if len(new) >= self.max_prototypes:
                    break
                if size < self.min_count_to_keep:
                    continue
                new.append((int(sid), F.normalize(centroid, p=2, dim=-1), float(size)))

        # Transactional: if nothing survives filtering (a sparse epoch), keep the
        # previous bank rather than wiping accumulated memory to empty.
        if not new:
            return
        self.prototypes.zero_()
        self.counts.zero_()
        self.active_mask.fill_(False)
        self.prototype_series.fill_(-1)
        for slot, (sid, centroid, size) in enumerate(new):
            self.prototypes[slot] = centroid
            self.counts[slot] = size
            self.active_mask[slot] = True
            self.prototype_series[slot] = sid

    def _cluster_centroids(self, members: torch.Tensor, max_k: int):
        """Cluster unit vectors into <= max_k count-weighted centroids.

        Deterministic and order-invariant: the rows are canonicalised before
        clustering, and centroids are returned largest-cluster-first so bank slot
        assignment is stable run-to-run.
        """
        n = int(members.size(0))
        if n == 0:
            return [], []
        keys = members.detach().cpu().numpy()
        order = sorted(range(n), key=lambda i: keys[i].tobytes())
        members = members[torch.as_tensor(order, device=members.device)]

        # Never request more clusters than can hold min_count_to_keep members
        # each, so rare series (few candidates) still form one supported prototype
        # instead of splitting into singletons that all get filtered out.
        k = min(int(max_k), max(1, n // max(1, self.min_count_to_keep)))
        if k <= 1:
            return [members.mean(dim=0)], [n]

        labels = self._agglomerative_labels(members, k)
        clusters = []
        for cl in range(int(labels.max().item()) + 1):
            m = members[labels == cl]
            if m.size(0) == 0:
                continue
            clusters.append((m.mean(dim=0), int(m.size(0))))
        clusters.sort(
            key=lambda cs: (-cs[1], cs[0].detach().cpu().numpy().tobytes())
        )
        return [c for c, _ in clusters], [s for _, s in clusters]

    @staticmethod
    def _agglomerative_labels(members: torch.Tensor, k: int) -> torch.Tensor:
        """Deterministic agglomerative-clustering labels for unit vectors."""
        from sklearn.cluster import AgglomerativeClustering

        x = members.detach().cpu().to(torch.float64).numpy()
        labels = AgglomerativeClustering(
            n_clusters=k, metric="euclidean", linkage="ward"
        ).fit_predict(x)
        return torch.as_tensor(labels, device=members.device, dtype=torch.long)

    def get_active_prototypes(self) -> torch.Tensor:
        return self.prototypes[self.active_mask]

    def get_active_prototype_series_ids(self) -> torch.Tensor:
        return self.prototype_series[self.active_mask]

    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def series_histogram(self) -> Dict[str, int]:
        active_series = self.prototype_series[self.active_mask]
        hist: Dict[str, int] = {}
        for sid in active_series.tolist():
            key = self._id_to_series.get(int(sid), "unknown")
            hist[key] = hist.get(key, 0) + 1
        return hist

    def series_seen_histogram(self) -> Dict[str, float]:
        return {
            self._id_to_series.get(int(sid), "unknown"): float(count)
            for sid, count in self._series_seen_counts.items()
        }

    def missing_seen_series(self) -> list[str]:
        active_sids = set(self.prototype_series[self.active_mask].tolist())
        missing = [
            self._id_to_series.get(int(sid), "unknown")
            for sid in self._series_seen_counts
            if int(sid) not in active_sids
        ]
        return sorted(missing)


def select_active_conformer_candidates(
    z: torch.Tensor,
    alpha: torch.Tensor,
    y_pIC50: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    series_labels: Optional[Sequence[object]] = None,
    active_threshold: float = 7.0,
    top_m: int = 3,
    return_series: bool = False,
):
    """Select top-attended conformers from active training bags.

    Parameters
    ----------
    z
        Contextualized conformer embeddings with shape [B, N, D].
    alpha
        Attention weights with shape [B, N].
    y_pIC50
        Continuous labels with shape [B].
    key_padding_mask
        Optional mask with True for padding.
    """
    if z.dim() == 2:
        z = z.unsqueeze(0)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)

    if z.dim() != 3:
        raise ValueError(f"Expected z with shape [B, N, D], got {tuple(z.shape)}")
    if alpha.dim() != 2:
        raise ValueError(
            f"Expected alpha with shape [B, N], got {tuple(alpha.shape)}"
        )
    if top_m <= 0:
        empty = z.new_zeros((0, z.size(-1)))
        return (empty, None) if return_series else empty

    y = y_pIC50.flatten().to(device=z.device)
    active_bags = y >= float(active_threshold)
    if active_bags.sum() == 0:
        empty = z.new_zeros((0, z.size(-1)))
        return (empty, None) if return_series else empty

    active_series = None
    if series_labels is not None:
        if len(series_labels) != z.size(0):
            raise ValueError(
                "series_labels length must match batch size: "
                f"{len(series_labels)} != {z.size(0)}"
            )
        active_indices = torch.nonzero(active_bags, as_tuple=False).flatten()
        active_series = [
            str(series_labels[int(i)])
            for i in active_indices.detach().cpu().tolist()
        ]

    z_active = z[active_bags]
    alpha_active = alpha[active_bags].to(device=z.device, dtype=z.dtype)
    mask_active = None

    if key_padding_mask is not None:
        mask_active = key_padding_mask.to(device=z.device).bool()[active_bags]
        alpha_active = alpha_active.masked_fill(mask_active, -1.0)

    k = min(int(top_m), z_active.size(1))
    top_idx = torch.topk(
        alpha_active,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    D = z_active.size(-1)
    selected = torch.gather(
        z_active,
        dim=1,
        index=top_idx.unsqueeze(-1).expand(-1, -1, D),
    )
    selected_series = None
    if active_series is not None:
        selected_series = [
            active_series[b]
            for b in range(len(active_series))
            for _ in range(k)
        ]

    if mask_active is not None:
        valid_selected = ~torch.gather(mask_active, dim=1, index=top_idx)
        if selected_series is not None:
            valid_flat = valid_selected.reshape(-1).detach().cpu().tolist()
            selected_series = [
                series
                for series, keep in zip(selected_series, valid_flat)
                if keep
            ]
        selected = selected[valid_selected]
    else:
        selected = selected.reshape(-1, D)

    if selected.numel() == 0:
        empty = z.new_zeros((0, D))
        return (empty, None) if return_series else empty

    candidates = selected.reshape(-1, D)
    if return_series:
        return candidates, selected_series
    return candidates


class ActivePrototypeQuery(nn.Module):
    """Build bag-specific cross-attention queries from active prototypes."""

    def __init__(
        self,
        dim: int = 256,
        top_m_prototypes: int = 8,
        temperature: float = 0.2,
        prefer_same_series: bool = True,
        fallback_to_all_series: bool = True,
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.top_m_prototypes = int(top_m_prototypes)
        self.temperature = float(temperature)
        self.prefer_same_series = bool(prefer_same_series)
        self.fallback_to_all_series = bool(fallback_to_all_series)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.top_m_prototypes <= 0:
            raise ValueError(
                f"top_m_prototypes must be positive, got {top_m_prototypes}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.fallback_query = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(2 * self.dim),
            nn.Linear(2 * self.dim, self.dim),
            nn.Tanh(),
        )

    @staticmethod
    def masked_mean(
        z: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if key_padding_mask is None:
            return z.mean(dim=1)

        valid = (~key_padding_mask.bool()).to(device=z.device, dtype=z.dtype)
        valid = valid.unsqueeze(-1)
        return (z * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(eps)

    def forward(
        self,
        z: torch.Tensor,
        prototype_bank: DynamicActivePrototypeBank,
        key_padding_mask: Optional[torch.Tensor] = None,
        series_labels: Optional[Sequence[object]] = None,
    ):
        """Return an active-prototype-informed query with shape [B, 1, D]."""
        if z.dim() != 3:
            raise ValueError(f"Expected z with shape [B, N, D], got {tuple(z.shape)}")

        B, _, D = z.shape
        if D != self.dim:
            raise ValueError(
                f"ActivePrototypeQuery dim={self.dim} received embeddings with D={D}"
            )

        r0 = self.masked_mean(z, key_padding_mask)
        P = prototype_bank.get_active_prototypes()

        if P.numel() == 0:
            query = self.fallback_query.to(device=z.device, dtype=z.dtype)
            return query.expand(B, -1, -1), {
                "proto_weights": None,
                "num_active_prototypes": 0,
            }

        P = P.to(device=z.device, dtype=z.dtype)
        r0_n = F.normalize(r0, p=2, dim=-1)
        P_n = F.normalize(P, p=2, dim=-1)
        logits = (r0_n @ P_n.T) / self.temperature
        same_series_available = None
        top_mask = None
        series_filter_used = False

        if self.prefer_same_series and series_labels is not None:
            if len(series_labels) != B:
                raise ValueError(
                    "series_labels length must match batch size for active "
                    f"prototype query: {len(series_labels)} != {B}"
                )
            proto_series = prototype_bank.get_active_prototype_series_ids().to(
                device=z.device,
            )
            bag_series = prototype_bank.series_ids_for_labels(
                series_labels,
                create=False,
                device=z.device,
            )
            same_series = (
                (bag_series.unsqueeze(1) >= 0)
                & (proto_series.unsqueeze(0) >= 0)
                & (bag_series.unsqueeze(1) == proto_series.unsqueeze(0))
            )
            same_series_available = same_series.any(dim=1)
            if same_series_available.any():
                if not self.fallback_to_all_series and not same_series_available.all():
                    missing = torch.nonzero(
                        ~same_series_available,
                        as_tuple=False,
                    ).flatten()
                    raise ValueError(
                        "No active prototypes for some requested series; "
                        f"first missing batch rows: {missing[:10].tolist()}"
                    )
                low = torch.finfo(logits.dtype).min
                logits_same = logits.masked_fill(~same_series, low)
                logits = torch.where(
                    same_series_available.unsqueeze(1),
                    logits_same,
                    logits,
                )
                series_filter_used = True

        K = P.size(0)
        m = min(self.top_m_prototypes, K)
        topv, topi = torch.topk(logits, k=m, dim=-1)
        if same_series_available is not None and series_filter_used:
            proto_series = prototype_bank.get_active_prototype_series_ids().to(
                device=z.device,
            )
            bag_series = prototype_bank.series_ids_for_labels(
                series_labels or [],
                create=False,
                device=z.device,
            )
            same_series = (
                (bag_series.unsqueeze(1) >= 0)
                & (proto_series.unsqueeze(0) >= 0)
                & (bag_series.unsqueeze(1) == proto_series.unsqueeze(0))
            )
            top_mask = same_series.gather(1, topi)
            top_mask = torch.where(
                same_series_available.unsqueeze(1),
                top_mask,
                torch.ones_like(top_mask),
            )
            topv = topv.masked_fill(~top_mask, torch.finfo(topv.dtype).min)
        weights_top = torch.softmax(topv, dim=-1)
        if top_mask is not None:
            weights_top = weights_top.masked_fill(~top_mask, 0.0)
            weights_top = weights_top / weights_top.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        P_selected = P[topi]
        active_context = (weights_top.unsqueeze(-1) * P_selected).sum(dim=1)

        query = self.query_proj(torch.cat([r0, active_context], dim=-1))
        return query.unsqueeze(1), {
            "proto_top_idx": topi,
            "proto_top_weights": weights_top,
            "num_active_prototypes": K,
            "proto_series_filter_used": series_filter_used,
            "proto_same_series_available": same_series_available,
        }
