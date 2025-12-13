#!/usr/bin/env python3
"""
Prepare potato experiment splits and checks for the partial-domain problem.
Writes `data/splits/potato_experiment.json` with recommended splits and warnings.

Usage:
    python scripts/prepare_potato_experiment.py

This script is conservative: it does not modify images, only reads
`data/processed/combined_dataset.csv` and writes a JSON summary.
"""
import csv, json, os, random

COMBINED = os.path.join('data','processed','combined_dataset.csv')
OUTDIR = os.path.join('data','splits')
OUTFILE = os.path.join(OUTDIR,'potato_experiment.json')

random.seed(42)

entries = {'plantvillage': [], 'plantdoc': []}
if not os.path.exists(COMBINED):
    raise SystemExit(f'Error: combined dataset not found: {COMBINED}')

with open(COMBINED,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 6:
            continue
        filename, label, crop, subclass, source, path = (
            row[0].strip(), row[1].strip(), row[2].strip().lower(), row[3].strip(), row[4].strip().lower(), row[5].strip()
        )
        if crop == 'potato':
            entries.setdefault(source, []).append({'filename': filename, 'label': label, 'subclass': subclass, 'path': path})

summary = {
    'total_potato': sum(len(v) for v in entries.values()),
    'counts': {k: len(v) for k, v in entries.items()},
    'classes_per_source': {},
}
for src, lst in entries.items():
    classes = {}
    for r in lst:
        classes[r['label']] = classes.get(r['label'], 0) + 1
    summary['classes_per_source'][src] = classes

# Recommendations and checks
# Default experiment budget/rounds from your master protocol: budget=10 rounds=5
master_budget = 10
master_rounds = 5
recommended = {}

pv_count = summary['counts'].get('plantvillage', 0)
pd_count = summary['counts'].get('plantdoc', 0)

recommended['note'] = (
    'PlantVillage (source) is large ({} samples); PlantDoc (target) is tiny ({} samples). '
    'This is the partial-domain case: PlantDoc is missing `potato_healthy` class.'
).format(pv_count, pd_count)

# pool size check
pool_size = pd_count
required = master_budget * master_rounds
warnings = []
if pool_size < required:
    warnings.append(f'TARGET POOL TOO SMALL for master protocol (pool={pool_size} < budget*rounds={required}).')

# classes missing in plantdoc
pv_classes = set(summary['classes_per_source'].get('plantvillage', {}).keys())
pd_classes = set(summary['classes_per_source'].get('plantdoc', {}).keys())
missing_in_pd = sorted(list(pv_classes - pd_classes))
if missing_in_pd:
    warnings.append(f'CLASSES MISSING IN PLANTDOC: {missing_in_pd}')

# Suggested actions
actions = [
    'Proceed but reduce budget/rounds (e.g., budget=2-3, rounds=2-3) to match available plantdoc images.',
    'Collect more field images for potato (recommended) so SSL and AL have sufficient unlabeled pool.',
    'Use tomato or pepper where PlantDoc has many target images if you want robust experiments now.',
    'Apply partial-domain adaptation methods; be cautious with FixMatch on very small target pool (may not help).',
]

# Build recommended split object (do not sample if plantdoc tiny)
# We'll recommend: source_train = all PlantVillage potato paths; target_pool = all PlantDoc potato paths
source_train = [r['path'] for r in entries.get('plantvillage', [])]
target_pool = [r['path'] for r in entries.get('plantdoc', [])]

# Optionally build a small seed labeled set from PlantDoc if there are >0 images
seed_labels = []
if len(target_pool) > 0:
    # create a tiny seed set (up to 5 images, stratified if possible)
    by_label = {}
    for r in entries.get('plantdoc', []):
        by_label.setdefault(r['label'], []).append(r)
    for lbl, items in by_label.items():
        seed_labels.extend([it['path'] for it in items[:1]])
    # cap seed to 5
    seed_labels = seed_labels[:5]

os.makedirs(OUTDIR, exist_ok=True)
with open(OUTFILE, 'w', encoding='utf-8') as f:
    json.dump({
        'summary': summary,
        'recommended': {
            'master_budget': master_budget,
            'master_rounds': master_rounds,
            'required_pool_size': required,
            'pool_size': pool_size,
            'warnings': warnings,
            'actions': actions,
            'source_train_count': len(source_train),
            'target_pool_count': len(target_pool),
            'seed_labels_suggested': seed_labels,
        },
        'source_train_paths': source_train,
        'target_pool_paths': target_pool
    }, f, indent=2)

print('WROTE', OUTFILE)
print('Summary:', json.dumps(summary, indent=2))
if warnings:
    print('\nWARNINGS:')
    for w in warnings:
        print('-', w)

