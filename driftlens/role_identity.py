"""Conservative CPU identity hypotheses anchored by an attributed seed pair.

Scores are histogram similarities, never probabilities or expert validation.
This module assigns roles to existing observed detections, never makes boxes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class IdentitySettings:
    min_similarity: float = .90
    min_role_margin: float = .04
    min_candidate_margin: float = .045
    min_confidence: float = .20
    min_crop_dimension: int = 14
    min_chromatic_fraction: float = .015
    max_pair_overlap_fraction: float = .45
    max_seed_overlap_fraction: float = .08
    max_exemplars_per_role: int = 24
    min_track_support_samples: int = 3
    neutral_saturation_ceiling: int = 60
    min_consensus_similarity: float = .93
    min_consensus_seed_similarity: float = .84
    max_consensus_anchor_gap_seconds: float = 1.5
    hue_weight: float = .12

    def __post_init__(self):
        for name in ('min_similarity', 'min_role_margin', 'min_candidate_margin',
                     'min_confidence', 'min_chromatic_fraction',
                     'max_pair_overlap_fraction', 'max_seed_overlap_fraction', 'hue_weight'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite value between zero and one.')
        for name in ('min_consensus_similarity', 'min_consensus_seed_similarity'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite value between zero and one.')
        if not math.isfinite(self.max_consensus_anchor_gap_seconds) or self.max_consensus_anchor_gap_seconds <= 0:
            raise ValueError('Consensus anchor age must be a positive finite number.')
        for name in ('min_crop_dimension', 'max_exemplars_per_role'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError('Crop size and exemplar count must be integers of at least two.')
        if isinstance(self.min_track_support_samples, bool) or not isinstance(self.min_track_support_samples, int) or self.min_track_support_samples < 1:
            raise ValueError('Track support must be a positive integer sample count.')
        if isinstance(self.neutral_saturation_ceiling, bool) or not isinstance(self.neutral_saturation_ceiling, int) or not 1 <= self.neutral_saturation_ceiling <= 255:
            raise ValueError('Neutral saturation ceiling must be an HSV integer from one to255.')


def _number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite.') from error
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite.')
    return number


def _identity(value):
    if value is None or str(value).strip().lower() in {'', 'none', 'nan', 'null'}:
        return None
    result = _number(value, 'Track ID')
    if isinstance(value, bool) or result < 0 or not result.is_integer():
        raise ValueError('Track IDs must be nonnegative integers.')
    return int(result)


def _overlap_fraction(left, right):
    """Intersection over the smaller box catches enclosing/merged pairs."""
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (left, right)]
    return intersection / max(1., min(areas))


def describe_crop(crop, settings=None):
    """View-tolerant colour mass plus chromatic livery hue distribution."""
    settings = settings or IdentitySettings()
    if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        return None
    if min(crop.shape[:2]) < settings.min_crop_dimension:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    # Mild blue or warm exposure casts on neutral body panels are not livery.
    def histogram(ceiling):
        chromatic = (saturation >= ceiling) & (value >= 45)
        dark = (value < 105) & ~chromatic
        white = (value >= 170) & (saturation < ceiling)
        gray = ~(chromatic | dark | white)
        masses = np.zeros(15, dtype=np.float64)
        for index in range(12):
            masses[index] = np.mean(chromatic & (hue >= index * 15) & (hue < (index + 1) * 15))
        masses[12:] = dark.mean(), gray.mean(), white.mean()
        chromatic_fraction = float(masses[:12].sum())
        hues = masses[:12] / max(chromatic_fraction, 1e-12)
        return {'masses': masses, 'hues': hues, 'chromatic_fraction': chromatic_fraction}
    primary = histogram(settings.neutral_saturation_ceiling)
    primary['neutral_cast'] = histogram(max(settings.neutral_saturation_ceiling, 100))
    return primary


def appearance_similarity(left, right, settings=None):
    settings = settings or IdentitySettings()
    def compare(a, b):
        mass_score = float(np.sqrt(a['masses'] * b['masses']).sum())
        hue_score = float(np.sqrt(a['hues'] * b['hues']).sum())
        hue_reliability = min(1., min(a['chromatic_fraction'], b['chromatic_fraction']) /
                              max(settings.min_chromatic_fraction, 1e-12))
        weight = settings.hue_weight * hue_reliability
        return (1 - weight) * mass_score + weight * hue_score
    return max(compare(left, right), compare(left.get('neutral_cast', left), right.get('neutral_cast', right)))


def _prototype_score(feature, exemplars, settings):
    scores = sorted((appearance_similarity(feature, exemplar, settings) for exemplar in exemplars), reverse=True)
    # Require support from several confirmed examples; a single outlier cannot win.
    return float(np.mean(scores[:min(3, len(scores))]))


def _geometry_continues(candidate, neighbours, clip_time):
    close = [(abs(time - clip_time), row) for time, row in neighbours
             if 0 < abs(time - clip_time) <= .35]
    if not close:
        return False
    other = min(close, key=lambda item: item[0])[1]
    box, previous = candidate['box'], other['box']
    width, height = box[2] - box[0], box[3] - box[1]
    pw, ph = previous[2] - previous[0], previous[3] - previous[1]
    area_ratio, shape_ratio = width * height / (pw * ph), (width / height) / (pw / ph)
    center_distance = math.hypot((box[0] + box[2] - previous[0] - previous[2]) / 2,
                                 (box[1] + box[3] - previous[1] - previous[3]) / 2)
    return .55 <= area_ratio <= 1.8 and .70 <= shape_ratio <= 1.4 and center_distance <= 1.5 * max(width, pw)


def _local_consensus(decisions, by_frame, features, exemplars, settings):
    """One frozen pass over strong role evidence; never iteratively grows banks."""
    banks, observed_tracks = defaultdict(list), defaultdict(list)
    for decision in decisions:
        for candidate in by_frame[decision['frame_index']]:
            observed_tracks[(decision['shot_index'], candidate['track_id'])].append((decision['clip_time'], candidate))
        for role in ('lead', 'chase'):
            identity = decision[f'{role}_id']
            if identity is None:
                continue
            feature = features.get((decision['frame_index'], identity))
            if feature is not None:
                banks[(decision['shot_index'], identity, role)].append((decision['clip_time'], feature))
    frozen = {}
    for key, values in banks.items():
        if len(values) < settings.min_track_support_samples:
            continue
        # Strong support must occur together, not merely accumulate isolated guesses.
        times = [item[0] for item in values]
        if not any(times[index + settings.min_track_support_samples - 1] - times[index] <= .5
                   for index in range(len(times) - settings.min_track_support_samples + 1)):
            continue
        indices = np.linspace(0, len(values) - 1, min(len(values), 24)).round().astype(int)
        frozen[key] = [values[index] for index in indices]
    proposals = {}
    for decision in decisions:
        if decision['assignment_kind'] != 'automatic_appearance':
            continue
        candidates = by_frame[decision['frame_index']]
        for role, other in (('lead', 'chase'), ('chase', 'lead')):
            if decision[f'{role}_id'] is not None:
                continue
            eligible = []
            for candidate in candidates:
                key = (decision['shot_index'], candidate['track_id'], role)
                anchors = frozen.get(key)
                feature = features.get((decision['frame_index'], candidate['track_id']))
                if anchors is None or feature is None or candidate['confidence'] < settings.min_confidence:
                    continue
                if min(abs(time - decision['clip_time']) for time, _ in anchors) > settings.max_consensus_anchor_gap_seconds:
                    continue
                if not _geometry_continues(candidate, observed_tracks[key[:2]], decision['clip_time']):
                    continue
                bank = [item[1] for item in anchors]
                score = _prototype_score(feature, bank, settings)
                seed_score = _prototype_score(feature, exemplars[role], settings)
                rival_score = _prototype_score(feature, exemplars[other], settings)
                if score < settings.min_consensus_similarity or seed_score < settings.min_consensus_seed_similarity or rival_score - seed_score > .015:
                    continue
                competitor_scores = [_prototype_score(other_feature, bank, settings)
                                     for rival in candidates if rival['track_id'] != candidate['track_id']
                                     and rival['confidence'] >= settings.min_confidence
                                     and (other_feature := features.get((decision['frame_index'], rival['track_id']))) is not None]
                margin = score - max(competitor_scores, default=0.)
                if margin < settings.min_candidate_margin:
                    continue
                eligible.append((score, candidate, seed_score, seed_score - rival_score, margin))
            if len(eligible) != 1:
                continue
            score, candidate, seed_score, role_margin, margin = eligible[0]
            if decision[f'{other}_id'] == candidate['track_id']:
                continue
            proposals[(decision['frame_index'], role)] = (candidate['track_id'], score, seed_score, role_margin, margin)
    added = 0
    for decision in decisions:
        for role in ('lead', 'chase'):
            proposal = proposals.get((decision['frame_index'], role))
            if proposal is None:
                continue
            identity, score, seed_score, role_margin, margin = proposal
            decision[f'{role}_id'] = identity
            decision[f'{role}_similarity'] = round(seed_score, 6)
            decision[f'{role}_role_margin'] = round(role_margin, 6)
            decision[f'{role}_candidate_margin'] = round(margin, 6)
            decision[f'{role}_consensus_similarity'] = round(score, 6)
            decision[f'{role}_reason'] = 'appearance_and_frozen_local_track_consensus'
            added += 1
        selected = [next((row for row in by_frame[decision['frame_index']] if row['track_id'] == decision[f'{role}_id']), None)
                    for role in ('lead', 'chase')]
        if all(selected) and (selected[0]['track_id'] == selected[1]['track_id'] or
                              _overlap_fraction(selected[0]['box'], selected[1]['box']) > settings.max_pair_overlap_fraction):
            decision['lead_id'] = decision['chase_id'] = None
            decision['lead_reason'] = decision['chase_reason'] = 'overlapping_or_merged_pair_boxes'
    return len(frozen), added


def _read_features(source_path, frames, by_frame, settings):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError('The saved source video cannot be opened for identity matching.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('The saved source video has no usable frame rate.')
    features = {}
    next_index = None
    try:
        for frame in frames:
            source_index = int(round(frame['source_time'] * fps))
            if next_index is None:
                if source_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                next_index = source_index
            if source_index < next_index:
                raise ValueError('Sampled source frame times must be strictly ordered.')
            while next_index < source_index:
                if not cap.grab():
                    raise ValueError('The source ended before the saved sampled frame.')
                next_index += 1
            okay, image = cap.read()
            next_index += 1
            if not okay:
                raise ValueError('A saved sampled source frame could not be decoded.')
            for candidate in by_frame[frame['frame_index']]:
                box = candidate['box']
                x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
                x2, y2 = min(image.shape[1], int(math.ceil(box[2]))), min(image.shape[0], int(math.ceil(box[3])))
                feature = describe_crop(image[y1:y2, x1:x2], settings)
                if feature is not None:
                    features[(frame['frame_index'], candidate['track_id'])] = feature
    finally:
        cap.release()
    return features


def compress_decisions(decisions, duration_seconds):
    """Compress contiguous decisions, splitting even identical IDs at camera cuts."""
    intervals = []
    for index, decision in enumerate(decisions):
        end = decisions[index + 1]['clip_time'] if index + 1 < len(decisions) else duration_seconds
        if end <= decision['clip_time']:
            raise ValueError('Decision timestamps must increase inside the clip.')
        key = (decision['lead_id'], decision['chase_id'], decision['shot_index'], decision['assignment_kind'])
        if intervals and intervals[-1]['_key'] == key:
            intervals[-1]['end_clip_seconds'] = end
        else:
            if decision['assignment_kind'] == 'user_seed':
                basis = 'User selected seed pair; observed detections only.'
            elif decision['assignment_kind'] == 'automatic_seed':
                basis = 'Automatically proposed initial pair and travel order; unverified hypothesis, not a user or expert role review.'
            else:
                basis = 'Automatic appearance hypothesis anchored to the selected pair; uncertain observations withheld; no expert validation.'
            intervals.append({'start_clip_seconds': decision['clip_time'], 'end_clip_seconds': end,
                              'lead_id': decision['lead_id'], 'chase_id': decision['chase_id'],
                              'review_basis': basis, 'assignment_kind': decision['assignment_kind'], '_key': key})
    for interval in intervals:
        interval.pop('_key')
    return intervals


def suggest_role_intervals(source_path, frames, observations, seed_intervals, duration_seconds, settings=None, *, seed_origin='user_reviewed'):
    """Use immutable user seed exemplars to reconnect observed vehicle roles.

    Each nonseed sample is checked independently, so track reuse, fragmentation,
    camera changes and livery changes cannot silently inherit a previous role.
    Global seed exemplars remain immutable. One frozen pass of strongly matched
    local observations may support weaker current crops, without iterative updates.
    """
    started = perf_counter()
    settings = settings or IdentitySettings()
    if isinstance(settings, dict):
        settings = IdentitySettings(**settings)
    if seed_origin not in {'user_reviewed', 'automatic_motion'}:
        raise ValueError('Seed origin must be user_reviewed or automatic_motion.')
    duration = _number(duration_seconds, 'Duration')
    if duration <= 0 or not frames:
        raise ValueError('Identity matching needs a positive duration and sampled frames.')
    canonical_frames = []
    for index, frame in enumerate(frames):
        number = _identity(frame.get('frame_index'))
        time = _number(frame.get('clip_time'), 'Frame time')
        source = _number(frame.get('source_time'), 'Source time')
        shot = _identity(frame.get('shot_index', 0))
        if number != index or shot is None or not 0 <= time < duration or source < 0:
            raise ValueError('Saved sampled frames must have ordered indices and valid times.')
        if canonical_frames and (time <= canonical_frames[-1]['clip_time'] or source <= canonical_frames[-1]['source_time']):
            raise ValueError('Saved sample times must strictly increase.')
        canonical_frames.append({'frame_index': number, 'clip_time': time, 'source_time': source, 'shot_index': shot})
    seeds = []
    for interval in seed_intervals:
        start = _number(interval.get('start_clip_seconds'), 'Seed start')
        end = _number(interval.get('end_clip_seconds'), 'Seed end')
        lead, chase = _identity(interval.get('lead_id')), _identity(interval.get('chase_id'))
        if not 0 <= start < end <= duration or lead is None or chase is None or lead == chase:
            raise ValueError('Each seed needs two distinct selected IDs and valid clip bounds.')
        if seeds and start < seeds[-1]['end_clip_seconds']:
            raise ValueError('Seed intervals must be ordered and cannot overlap.')
        seeds.append({'start_clip_seconds': start, 'end_clip_seconds': end, 'lead_id': lead, 'chase_id': chase})
    if not seeds:
        raise ValueError('Select a reviewed lead and chase seed interval first.')
    by_frame = defaultdict(list)
    seen = set()
    for observation in observations:
        if str(observation.get('observed', True)).lower() not in {'true', '1', '1.0', 'yes'}:
            continue
        index, identity = _identity(observation.get('frame_index')), _identity(observation.get('track_id'))
        if identity is None:
            continue
        if index is None or index >= len(canonical_frames) or (index, identity) in seen:
            raise ValueError('Observations need a unique current track and saved sampled frame.')
        seen.add((index, identity))
        frame = canonical_frames[index]
        if observation.get('clip_time') is not None and abs(_number(observation['clip_time'], 'Observation time') - frame['clip_time']) > .001:
            raise ValueError('An observation time disagrees with its saved frame.')
        if _identity(observation.get('shot_index', 0)) != frame['shot_index']:
            raise ValueError('An observation camera view disagrees with its saved frame.')
        box = [_number(observation.get(name), 'Box coordinate') for name in ('x1', 'y1', 'x2', 'y2')]
        confidence = _number(observation.get('confidence', 0), 'Detection confidence')
        if box[2] <= box[0] or box[3] <= box[1] or not 0 <= confidence <= 1:
            continue
        by_frame[index].append({'track_id': identity, 'box': box, 'confidence': confidence})
    features = _read_features(source_path, canonical_frames, by_frame, settings)
    exemplars = {'lead': [], 'chase': []}
    seed_for_frame = {}
    for frame in canonical_frames:
        seed = next((item for item in seeds if item['start_clip_seconds'] <= frame['clip_time'] < item['end_clip_seconds']), None)
        if seed is None:
            continue
        seed_for_frame[frame['frame_index']] = seed
        candidates = {row['track_id']: row for row in by_frame[frame['frame_index']]}
        pair = [candidates.get(seed[f'{role}_id']) for role in ('lead', 'chase')]
        if any(candidate is None or candidate['confidence'] < settings.min_confidence for candidate in pair):
            continue
        if _overlap_fraction(pair[0]['box'], pair[1]['box']) > settings.max_seed_overlap_fraction:
            continue
        pair_features = [features.get((frame['frame_index'], candidate['track_id'])) for candidate in pair]
        if any(feature is None for feature in pair_features):
            continue
        for role, feature in zip(('lead', 'chase'), pair_features):
            exemplars[role].append(feature)
    if min(map(len, exemplars.values())) < 3:
        raise ValueError('The selected seed needs at least three clear, separated sampled observations of both cars.')
    original_counts = {role: len(bank) for role, bank in exemplars.items()}
    for role, bank in exemplars.items():
        indices = np.linspace(0, len(bank) - 1, min(len(bank), settings.max_exemplars_per_role)).round().astype(int)
        exemplars[role] = [bank[index] for index in indices]
    seed_similarity = max(appearance_similarity(left, right, settings) for left in exemplars['lead'] for right in exemplars['chase'])
    if seed_similarity > .96:
        raise ValueError('The selected pair has too similar an appearance for reliable livery matching. Use manual role intervals or a clearer seed view.')
    decisions = []
    for frame in canonical_frames:
        decision = {**frame, 'lead_id': None, 'chase_id': None, 'lead_similarity': None,
                    'chase_similarity': None, 'lead_role_margin': None, 'chase_role_margin': None,
                    'lead_candidate_margin': None, 'chase_candidate_margin': None,
                    'lead_consensus_similarity': None, 'chase_consensus_similarity': None,
                    'lead_reason': 'no_eligible_observation', 'chase_reason': 'no_eligible_observation',
                    'assignment_kind': 'automatic_appearance'}
        candidates = by_frame[frame['frame_index']]
        seed = seed_for_frame.get(frame['frame_index'])
        if seed:
            decision['assignment_kind'] = 'user_seed' if seed_origin == 'user_reviewed' else 'automatic_seed'
            for role in ('lead', 'chase'):
                if any(candidate['track_id'] == seed[f'{role}_id'] for candidate in candidates):
                    decision[f'{role}_id'] = seed[f'{role}_id']
                    decision[f'{role}_reason'] = 'user_selected_seed' if seed_origin == 'user_reviewed' else 'automatically_proposed_seed'
                else:
                    decision[f'{role}_reason'] = 'selected_seed_not_observed'
            pair = [next((candidate for candidate in candidates if candidate['track_id'] == decision[f'{role}_id']), None) for role in ('lead', 'chase')]
            if all(pair) and _overlap_fraction(pair[0]['box'], pair[1]['box']) > settings.max_pair_overlap_fraction:
                decision['lead_id'] = decision['chase_id'] = None
                decision['lead_reason'] = decision['chase_reason'] = 'overlapping_or_merged_seed_boxes'
            decisions.append(decision)
            continue
        scored = []
        for candidate in candidates:
            feature = features.get((frame['frame_index'], candidate['track_id']))
            if candidate['confidence'] < settings.min_confidence or feature is None:
                continue
            row = {**candidate, 'lead': _prototype_score(feature, exemplars['lead'], settings),
                   'chase': _prototype_score(feature, exemplars['chase'], settings)}
            scored.append(row)
        chosen = {}
        for role, other in (('lead', 'chase'), ('chase', 'lead')):
            ranked = sorted(scored, key=lambda candidate: (-candidate[role], candidate['track_id']))
            if not ranked:
                continue
            best = ranked[0]
            role_margin = best[role] - best[other]
            candidate_margin = best[role] - ranked[1][role] if len(ranked) > 1 else best[role]
            decision[f'{role}_similarity'] = round(best[role], 6)
            decision[f'{role}_role_margin'] = round(role_margin, 6)
            decision[f'{role}_candidate_margin'] = round(candidate_margin, 6)
            if best[role] < settings.min_similarity:
                reason = 'appearance_below_threshold'
            elif role_margin < settings.min_role_margin:
                reason = 'role_appearance_ambiguous'
            elif candidate_margin < settings.min_candidate_margin:
                reason = 'multiple_similar_candidates'
            else:
                chosen[role] = best
                decision[f'{role}_id'] = best['track_id']
                reason = 'appearance_matches_selected_role'
            decision[f'{role}_reason'] = reason
        if len(chosen) == 2 and (chosen['lead']['track_id'] == chosen['chase']['track_id'] or
                                _overlap_fraction(chosen['lead']['box'], chosen['chase']['box']) > settings.max_pair_overlap_fraction):
            decision['lead_id'] = decision['chase_id'] = None
            decision['lead_reason'] = decision['chase_reason'] = 'overlapping_or_merged_pair_boxes'
        decisions.append(decision)
    # A track that alternates roles is an identity warning, not grounds to fill
    # weak observations. Only suppress conflicting automatic assignments.
    track_roles = defaultdict(lambda: {'lead': 0, 'chase': 0})
    for decision in decisions:
        for role in ('lead', 'chase'):
            identity = decision[f'{role}_id']
            if identity is not None:
                track_roles[(decision['shot_index'], identity)][role] += 1
    for decision in decisions:
        if decision['assignment_kind'] in {'user_seed', 'automatic_seed'}:
            continue
        for role in ('lead', 'chase'):
            identity = decision[f'{role}_id']
            if identity is None:
                continue
            counts = track_roles[(decision['shot_index'], identity)]
            other = 'chase' if role == 'lead' else 'lead'
            if counts[other] >= 3 and counts[other] > counts[role]:
                decision[f'{role}_id'] = None
                decision[f'{role}_reason'] = 'conflicting_livery_on_local_track'
            elif counts[role] < settings.min_track_support_samples:
                decision[f'{role}_id'] = None
                decision[f'{role}_reason'] = 'insufficient_temporal_support'
    consensus_bank_count, consensus_proposals = _local_consensus(decisions, by_frame, features, exemplars, settings)
    intervals = compress_decisions(decisions, duration)
    automatic = [decision for decision in decisions if decision['assignment_kind'] == 'automatic_appearance']
    report = {'method': 'seed_livery_and_frozen_local_consensus_v2', 'settings': asdict(settings),
              'seed_intervals': seeds, 'seed_origin': seed_origin, 'seed_observation_counts': original_counts,
              'exemplar_counts': {role: len(bank) for role, bank in exemplars.items()},
              'maximum_cross_role_seed_similarity': round(seed_similarity, 6),
              'sampled_frames': len(decisions), 'automatic_frames': len(automatic),
              'automatic_lead_frames': sum(decision['lead_id'] is not None for decision in automatic),
              'automatic_chase_frames': sum(decision['chase_id'] is not None for decision in automatic),
              'automatic_paired_frames': sum(decision['lead_id'] is not None and decision['chase_id'] is not None for decision in automatic),
              'automatic_lead_unknown_frames': sum(decision['lead_id'] is None for decision in automatic),
              'automatic_chase_unknown_frames': sum(decision['chase_id'] is None for decision in automatic),
              'paired_frames': sum(decision['lead_id'] is not None and decision['chase_id'] is not None for decision in decisions),
              'frozen_local_consensus_banks': consensus_bank_count,
              'consensus_proposals_before_merged_box_veto': consensus_proposals,
              'accepted_consensus_role_frames': sum(decision[f'{role}_id'] is not None and decision[f'{role}_reason'] == 'appearance_and_frozen_local_track_consensus'
                                                     for decision in decisions for role in ('lead', 'chase')),
              'score_interpretation': 'Histogram similarity, not a correctness probability.',
              'validation_status': 'Algorithmic identity hypotheses require visual review; similar liveries and changed lighting can remain unknown or mismatch.',
              'exemplar_update_policy': 'Global seed templates remain immutable. A single frozen bank of strongly supported within-view observations may validate current boxes; consensus matches never update either bank.',
              'processing_seconds': round(perf_counter() - started, 3)}
    return {'intervals': intervals, 'decisions': decisions, 'summary': report}
