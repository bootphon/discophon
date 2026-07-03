import copy
from collections import defaultdict
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from pyannote.core import Annotation, Segment, Timeline

from discophon.data import read_rttm


def rttm_to_annotations(rttm: pl.DataFrame, *, target_type: str = "SPEAKER", n_jobs: int = -1) -> list[Annotation]:
    def aux(uri: str, turns: pl.DataFrame) -> Annotation:
        annotation = Annotation(uri=uri)
        for i, turn in enumerate(turns.iter_rows(named=True)):
            if turn["Type"] != target_type:
                continue
            segment = Segment(turn["Turn Onset"], turn["Turn Onset"] + turn["Turn Duration"])
            annotation[segment, i] = turn["Speaker Name"]
        return annotation

    annotations = Parallel(n_jobs=n_jobs)(delayed(aux)(uri, turns) for (uri,), turns in rttm.group_by("File ID"))
    return sorted(annotations, key=lambda x: x.uri)


def read_annotations(path: str | Path) -> list[Annotation]:
    rttm = read_rttm(path) if Path(path).is_file() else pl.concat([read_rttm(p) for p in Path(path).glob("*.rttm")])
    return rttm_to_annotations(rttm)


def map_segment_to_silences(original: Annotation, processed: Annotation) -> dict[Segment, Timeline]:
    """Maps each speech segment in the processed annotation to the silences
    occurring within the corresponding segment in the original annotation.
    """
    mapping = defaultdict(list)
    silences_in_processed = processed.extrude(Timeline(original.itersegments()))
    for (seg, _), (silence, _) in processed.co_iter(silences_in_processed):
        mapping[seg].append(silence)
    return {seg: Timeline(mapping[seg]) for seg in processed.itersegments()}


def split_by_silence(
    segment: Segment,
    silences: Timeline,
    min_duration_on: float | None,
    max_duration_on: float,
) -> Timeline:
    """Recursively split `segment` using the longest silence in `silences` until
    all resulting segments are shorter than `max_duration_on`. Segments shorter than
    `min_duration_on` are discarded."""
    timeline = Timeline([segment])
    if min_duration_on is not None and timeline.duration() <= min_duration_on:  # Remove too short segments
        return Timeline([])
    if timeline.duration() <= max_duration_on:  # Termination condition: the segment is short enough
        return timeline
    if not silences:  # No more silence to split the segment: cut in the middle
        left = split_by_silence(Segment(segment.start, segment.middle), silences, min_duration_on, max_duration_on)
        right = split_by_silence(Segment(segment.middle, segment.end), silences, min_duration_on, max_duration_on)
        return Timeline(left.segments_list_ + right.segments_list_)
    longest_hole = max(silences, key=lambda x: x.duration)  # Find the longest silence
    timeline = timeline.extrude(longest_hole)
    new_timelines = [split_by_silence(seg, silences.crop(seg), min_duration_on, max_duration_on) for seg in timeline]  # ty: ignore[invalid-argument-type]
    return Timeline([s for t in new_timelines for s in t.segments_list_])


def cut_long_segments(
    original: Annotation,
    active: Annotation,
    min_duration_on: float | None,
    max_duration_on: float,
) -> Annotation:
    new_segments, segment_to_silences = [], map_segment_to_silences(original, active)
    for seg in active.itersegments():
        silences = segment_to_silences[seg]
        new_segments += list(split_by_silence(seg, silences, min_duration_on, max_duration_on).segments_list_)
    annotation = Annotation(uri=original.uri)
    for i, segment in enumerate(new_segments):
        annotation[segment, i] = "SPEECH"
    return annotation


def remove_long_silences(annotation: Annotation, min_duration_off: float) -> Annotation:
    return annotation.support(collar=min_duration_off)


def remove_short_segments(annotation: Annotation, min_duration_on: float) -> Annotation:
    active = copy.deepcopy(annotation)
    for segment, track, *_ in list(active.itertracks()):
        if segment.duration < min_duration_on:
            del active[segment, track]
    return active


def post_process(
    original: Annotation,
    *,
    min_duration_on: float,
    min_duration_off: float,
    max_duration_on: float,
) -> Annotation:
    active = copy.deepcopy(original)
    if min_duration_off > 0.0:
        active = remove_long_silences(active, min_duration_off)
    if min_duration_on > 0.0:
        active = remove_short_segments(active, min_duration_on)
    if max_duration_on > 0.0:
        active = cut_long_segments(original, active, min_duration_on, max_duration_on)
    return active


def batch_post_process(
    source: str | Path,
    dest: str | Path,
    *,
    min_duration_on: float,
    min_duration_off: float,
    max_duration_on: float,
) -> None:
    annotations = Parallel(n_jobs=-1)(
        delayed(post_process)(
            annotation,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            max_duration_on=max_duration_on,
        )
        for annotation in read_annotations(source)
    )
    for annotation in annotations:
        with Path(dest).open("a", encoding="utf-8") as f:
            annotation.write_rttm(f)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to RTTM file or directory")
    parser.add_argument("dest", type=Path, help="Output RTTM file")
    parser.add_argument("--min-duration-on", type=float, help="Min duration of segments (in seconds)", default=0.5)
    parser.add_argument("--min-duration-off", type=float, help="Min duration of silences (in seconds)", default=2.0)
    parser.add_argument("--max-duration-on", type=float, help="Max duration of segments (in seconds).", default=30.0)
    args = parser.parse_args()
    batch_post_process(
        args.source,
        args.dest,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        max_duration_on=args.max_duration_on,
    )
