"""
Stage `cleaned` (recordings -> cleaned): removes the ABP artefacts and the outliers of ABP and CBFV, trims the gaps,
interpolates and smooths the signals: {condition}/{file}_PP.csv. BPM and ETCO2 are not cleaned, only trimmed at the
same positions as ABP and CBFV. Files that fail are copied to the spoiled data.

A recording is cleaned only if it has enough usable signal: the samples recorded at the paced breathing rate of its
condition (paced_breaths in settings.toml) within +-breath_tolerance breaths per minute; in the other conditions (BAS)
all samples. The usable samples must make up at least min_usable_fraction of the recording and min_usable_duration
seconds (at fs); otherwise the recording is copied to the spoiled data.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.stages.cleaned.cleaning_chain import PreprocessData


def count_usable(bpm: pd.Series, breaths: float | None, tolerance: float) -> int:
    """
    Function to count the usable samples of a recording: the ones with the breathing rate within the tolerance of the
    paced rate (a missing rate is not usable), or all samples if the condition is not paced.
    :param bpm: recorded breathing rate [breaths/min], one value per sample.
    :param breaths: paced breathing rate of the condition [breaths/min] (None: not paced).
    :param tolerance: allowed difference from the paced rate [breaths/min].
    :return: number of usable samples.
    """
    if breaths is None:
        return len(bpm)
    return int(((bpm - breaths).abs() <= tolerance).sum())


class CleanSignals(Stage):
    PARAMS = {"fs": int, "paced_breaths": dict, "breath_tolerance": float, "min_usable_fraction": float,
              "min_usable_duration": float}

    def run(self) -> None:
        ctx = self.ctx
        fs = ctx.params["fs"]
        paced_breaths = ctx.params["paced_breaths"]
        tolerance = ctx.params["breath_tolerance"]
        min_fraction, min_duration = ctx.params["min_usable_fraction"], ctx.params["min_usable_duration"]
        base_dir = ctx.input()
        for directory in os.listdir(base_dir):
            dir_path = base_dir / directory
            out_dir = ctx.output(directory)
            for file in os.listdir(dir_path):
                file_path = dir_path / file
                bpm = pd.read_csv(file_path, delimiter=';', usecols=['BPM'])['BPM']
                usable = count_usable(bpm, paced_breaths.get(directory), tolerance)
                fraction = usable / len(bpm) if len(bpm) else 0.0
                if fraction < min_fraction or usable / fs < min_duration:
                    ctx.reject(file_path, directory,
                               f"too little usable signal: {usable / fs:.1f} s of {len(bpm) / fs:.1f} s "
                               f"({fraction:.0%}) at the breathing rate of the condition")
                    continue
                try:
                    prep = PreprocessData(str(file_path), first_column='ABP', second_column='CBFV', third_column='BPM',
                                          fourth_column='ETCO2')
                    df = prep.get_preprocessed_data()
                except Exception as e:
                    ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                    continue
                df.to_csv(out_dir / f"{file_path.stem}_PP.csv", sep=';', index=False)
                ctx.processed(f"{directory}/{file}")
