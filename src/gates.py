"""Release gates that combine measured module outputs without altering images."""
import math
import numpy as np
from .quality import geometry_decision


def geometry_gate(metrics, instance_image, registry, cfg):
    decision = geometry_decision(metrics, cfg)
    minimum = cfg['quality']['instance_pixels_min']
    ids, counts = np.unique(instance_image, return_counts=True)
    counts = {str(int(i)): int(n) for i, n in zip(ids, counts) if int(i)}
    meaningful = {i: n for i, n in counts.items() if n >= minimum}
    morphologies = {}
    for semantic, required, kind in ((3, 3, 'coral'), (5, 2, 'fish')):
        visible = sorted({registry[i]['morphology'] for i in meaningful
                          if int(registry[i]['semantic_id']) == semantic})
        morphologies[kind] = visible
        if len(visible) < required:
            decision['reasons'].append('insufficient_meaningful_' + kind + '_morphologies')
    decision['meaningful_visibility'] = {
        'minimum_pixels': minimum, 'instance_pixels': counts,
        'meaningful_instance_pixels': meaningful, 'morphologies': morphologies,
        'fish_count': sum(int(registry[i]['semantic_id']) == 5 for i in meaningful),
        'definition': 'first-hit GT pixel area, after occlusion; active objects may have zero pixels',
    }
    decision['passed'] = not decision['reasons']
    return decision


def noise_gate(metrics, cfg):
    reasons = []
    for ratio_key, threshold_key, numerator, zero_flag in (
        ('effect_to_noise_ratio', 'effect_noise_min', 'effect_rms', 'noise_zero'),
        ('structure_noise_ratio', 'structure_noise_min', 'structure_contrast', 'structure_noise_zero'),
    ):
        value = metrics[ratio_key]
        if value is None:
            passed = bool(metrics[zero_flag]) and metrics[numerator] > 0
        else:
            passed = math.isfinite(value) and value >= cfg['quality'][threshold_key]
        if not passed:
            reasons.append(threshold_key + '_not_met')
    return {'passed': not reasons, 'reasons': reasons, 'metrics': metrics,
            'thresholds': {k: cfg['quality'][k] for k in ('effect_noise_min', 'structure_noise_min')}}
