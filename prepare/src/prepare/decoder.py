import functools
from typing import Iterable, Optional

import numpy as np
import panphon
from jaxtyping import Bool, Integer

PHON_FEAT_DIM = 22

SILENCE = "SIL"
ZERO_TONE = "˧"


class UniqueSegmentFeature:
    def __init__(self, segments: Iterable[str], sum_diphthong: bool):
        self.sum_diphthong = sum_diphthong
        self.ft = panphon.FeatureTable()

        mono_segment_feature, multi_segment_feature = self.stratify_segments(segments)

        self.features_to_segment: dict[tuple[int, ...], tuple[str, set[str]]] = {}
        self.unique_segments: set[str] = set()
        self.multithongs: set[str] = set()
        self.add_segment_feature_batch(mono_segment_feature)
        self.add_segment_feature_batch(multi_segment_feature)

    def stratify_segments(self, segments: Iterable[str]) -> tuple[
        list[tuple[str, list[tuple[int, ...]]]], list[tuple[str, list[tuple[int, ...]]]]
    ]:
        mono_segment_feature, multi_segment_feature = [], []
        for seg in segments:
            feats = self.ft.word_to_vector_list(seg, numeric=True)
            if len(feats) == 0:
                print(
                    "Warning: a phoneme was not found in panphon's feature table: "
                    f"'{seg}'. Assuming it to be a silence (zero vector)."
                )
                feats = [(0,) * PHON_FEAT_DIM]
            else:
                feats = [tuple(fs[:PHON_FEAT_DIM]) for fs in feats]
            if len(feats) == 1:
                mono_segment_feature.append((seg, feats))
            else:
                multi_segment_feature.append((seg, feats))
        return mono_segment_feature, multi_segment_feature

    def add_segment_feature(self, segment: str, features: tuple[int, ...]) -> None:
        if segment not in self.unique_segments:
            self.unique_segments.add(segment)
            try:
                _, eq_segments = self.features_to_segment[features]
                eq_segments.add(segment)
            except KeyError:
                self.features_to_segment[features] = (segment, {segment})

    def add_segment_feature_batch(
            self,
            segment_feature_batch: list[tuple[str, list[tuple[int, ...]]]]
    ) -> None:
        for segment, features in segment_feature_batch:
            if len(features) == 1:
                self.add_segment_feature(segment, features[0])
            elif self.sum_diphthong:
                self.add_segment_feature(
                    segment,
                    tuple(
                        feats[0] if len(set(feats)) == 1 else 0 for feats in
                        zip(*features)
                    )
                )
            else:
                assert len(segment) == len(features), \
                    f"Expected {len(features)} segments for '{segment}', but got {len(segment)}"
                self.multithongs.add(segment)
                for seg, feats in zip(segment, features):
                    self.add_segment_feature(seg, feats)


class UniversalUniqueSegmentFeature:
    def __init__(self, sum_diphthong: bool):
        self.sum_diphthong = sum_diphthong
        self.ft = panphon.FeatureTable()

        self.features_to_segment: dict[tuple[int, ...], tuple[str, set[str]]] = {}
        self.unique_segments: set[str] = set()
        self.multithongs: set[str] = set()
        for segment, features in self.ft.segments:
            self.add_segment_feature(segment, tuple(features.numeric()[:PHON_FEAT_DIM]))

    def add_segment_feature(self, segment: str, features: tuple[int, ...]) -> None:
        if segment not in self.unique_segments:
            self.unique_segments.add(segment)
            try:
                _, eq_segments = self.features_to_segment[features]
                eq_segments.add(segment)
            except KeyError:
                self.features_to_segment[features] = (segment, {segment})


class FeatureDecoder:
    def __init__(
            self, sum_diphthong: bool, lang_segments: Optional[Iterable[str]] = None
    ) -> None:
        self.fake_segments: dict[tuple[int, ...], str] = {}
        self._segment_to_representative: dict[str, tuple[str, ...]] = {}

        self.multilingual_mode = lang_segments is None
        if self.multilingual_mode:
            self.unique_seg_feats = UniversalUniqueSegmentFeature(sum_diphthong)
        else:
            assert sum_diphthong is not None
            self.unique_seg_feats = UniqueSegmentFeature(lang_segments, sum_diphthong)

        self._features = np.asarray(
            list(self.unique_seg_feats.features_to_segment.keys())
        )
        self._representative_to_feature = dict(zip(self.segments, self._features))

        for rep, eq_segments in self.unique_seg_feats.features_to_segment.values():
            for seg in eq_segments:
                self._segment_to_representative[seg] = (rep,)
        for seg in self.unique_seg_feats.multithongs:
            self._segment_to_representative[seg] = tuple(
                rep for s in seg for rep in self._segment_to_representative[s]
            )

        featuresT = self._features.T
        feature_to_indices: list[dict[int, Bool[np.ndarray, "feats"]]] = []
        for d in range(featuresT.shape[0]):
            feature_to_indices.append({-1: featuresT[d] <= 0, 1: featuresT[d] >= 0})
        self.feature_to_indices = feature_to_indices

    @functools.cached_property
    def header(self) -> tuple[str, ...]:
        return tuple(self.unique_seg_feats.ft.names[:PHON_FEAT_DIM])

    @functools.cached_property
    def segments(self) -> tuple[str, ...]:
        return tuple(
            rep for rep, _ in self.unique_seg_feats.features_to_segment.values()
        )

    @property
    def features(self) -> Integer[np.ndarray, "segments features"]:
        return self._features

    @functools.cached_property
    def zero_index(self) -> int:
        if self.multilingual_mode:
            for k, (_, segs) in enumerate(
                    self.unique_seg_feats.features_to_segment.values()
            ):
                if ZERO_TONE in segs:
                    return k
        else:
            return self.segments.index(SILENCE)

    def segment_to_representative(self, segment: str) -> tuple[str, ...]:
        if segment == SILENCE:
            return (self.segments[self.zero_index],)
        if segment in self._segment_to_representative:
            return self._segment_to_representative[segment]
        assert self.multilingual_mode, \
            (f"Unable to find segment {segment} in the segment list provided for the "
             f"language.")
        features = self.unique_seg_feats.ft.word_to_vector_list(segment, numeric=True)
        return tuple(
            self.unique_seg_feats.features_to_segment[tuple(feats[:PHON_FEAT_DIM])][0]
            for feats in features
        )

    def canonical_representation(self, representative: tuple[str, ...]) \
            -> tuple[tuple[str, ...], Integer[np.ndarray, "_ features"]]:
        features = [self._representative_to_feature[rep] for rep in representative]
        if self.unique_seg_feats.sum_diphthong and len(representative) > 1:
            assert self.multilingual_mode
            representative = ("".join(representative),)
            features = [np.asarray(
                [feats[0] if len(set(feats)) == 1 else 0 for feats in zip(*features)]
            )]
        return representative, np.stack(features, axis=0)

    def find_segment(self, features: tuple[int, ...]) -> str:
        try:
            return self.fake_segments[features]
        except KeyError:
            pass

        # if all the features are equal to 0, should return silence
        if not any(features):
            indices = [self.zero_index]
        else:
            indices = np.flatnonzero(
                np.logical_and.reduce(
                    [feat2idx[f] for feat2idx, f in
                     zip(self.feature_to_indices, features) if f != 0]
                )
            )

        if len(indices) == 1:
            return self.segments[indices[0]]
        # if there are several possible segments, return the one with the least 0s
        elif len(indices) > 1:
            indices = indices.tolist()
            indices.sort(key=lambda idx: np.sum(self.features[idx] == 0))
            return self.segments[indices[0]]

        return self.fake_segments.setdefault(features, str(len(self.fake_segments) + 1))
