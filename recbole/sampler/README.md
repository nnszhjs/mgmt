# recbole.sampler

The sampling module for RecBole. It provides various negative sampling strategies used during model training to generate negative examples (items that the user has not interacted with) for pairwise and pointwise learning objectives.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initializer; exports `Sampler`, `KGSampler`, `RepeatableSampler`, and `SeqSampler`. |
| `sampler.py` | Implements sampling classes: `AbstractSampler` (base class with uniform and popularity-based sampling via alias tables), `Sampler` (standard negative sampler for user-item interactions supporting dynamic negative sampling), `KGSampler` (negative sampler for knowledge graph triplets that corrupts tail entities), `RepeatableSampler` (ensures reproducible negative samples across evaluations), and `SeqSampler` (generates negative item sequences for sequential models like DIEN). |
