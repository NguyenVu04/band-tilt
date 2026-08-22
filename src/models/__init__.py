"""One module per model, each owning everything that genuinely differs.

A model module owns its preprocessing, feature engineering, architecture,
training loop and prediction logic. There is deliberately no shared
``features/`` package: transforms that only one model needs live with that
model, where they can change without disturbing anything else.

What a model module must NOT implement
--------------------------------------
============================  ============================  ======================================
Component                     Belongs in                    Why
============================  ============================  ======================================
Train/val/test splitting      ``src.data.split``            models on different folds can't compare
Evaluation metrics            ``src.evaluation.metrics``    divergent definitions void the results
============================  ============================  ======================================

Model resolution
----------------
Models are instantiated from config through Hydra's ``_target_`` mechanism::

    from hydra.utils import instantiate

    model = instantiate(cfg.models)

There is no registry module on purpose — ``_target_`` already maps names to
classes, and maintaining both would mean two places to update and one to
forget. Adding a model therefore means adding a module here plus a config in
``configs/models/``; nothing in this file needs to change.
"""
