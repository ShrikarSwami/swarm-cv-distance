"""ML swarm-reconstruction track package.

Owned modules (see ORCHESTRATOR_ML.md ownership map):
  scene_gen.py     T1 scene + camera generator (scene agent)
  render_harness   T2 resumable render harness (harness agent)
  control.py       T2 control-file helpers (harness agent)
  status_app.py    T3 status page (status agent)
  metrics.py       shared metrics, frozen on acceptance (T2 build order)
  pack_dataset.py  T4 packing + splits
  model.py/train.py / eval_sweep.py / recon_app.py  T6/T7/T9
"""
