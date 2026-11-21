"""Demand, participants and scenario boundaries."""
import json
from pathlib import Path


def scoring_scene(run_dir, scene_dir=None):
    """Read the routes and optional road measurements used for a run."""
    receipt = run_dir / 'input_receipt.json'
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        files = saved.get('scene_files', {})
        if 'candidate_routes.json' in files:
            return receipt, files['candidate_routes.json'], files.get('measurements.json')
    if scene_dir is None:
        raise ValueError(f'{run_dir}: no saved routes in input_receipt.json. Supply the original scene directory, or use rescore --use-suite-scenes with the original suite.')
    folder = Path(scene_dir)
    routes = folder / 'candidate_routes.json'
    measurements = folder / 'measurements.json'
    return routes, json.loads(routes.read_text()), json.loads(measurements.read_text()) if measurements.is_file() else None
