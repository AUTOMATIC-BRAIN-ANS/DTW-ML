"""
Stage `aligned` (normalised -> aligned): keeps only the subjects present in every condition of [analysis] conditions,
so that the DTW costs, their statistics and the classification compare the same subjects in all conditions:
{condition}/{file} (copied unchanged). A subject rejected by an earlier stage in one condition is copied from the other
conditions to the spoiled data ({condition}/aligning_data, as in version 1.0) with the conditions it is missing from.
The subject is the prefix of the file name before the first "_" (V12_B6_PP_F_M_PP.csv -> V12).
"""

import os
import shutil

from ddtw_ml.pipeline import PipelineError, Stage

# subdirectory of the spoiled data of a condition for the subjects removed here (the name used in version 1.0)
ALIGNING_DIR = "aligning_data"


def subject_id(file: str) -> str:
    return file.split('_')[0]


class AlignSubjects(Stage):
    def run(self) -> None:
        ctx = self.ctx
        base_dir = ctx.input()
        conditions = ctx.analysis["conditions"]
        missing_dirs = [c for c in conditions if not (base_dir / c).is_dir()]
        if missing_dirs:
            raise PipelineError(f"No input of the conditions {missing_dirs} in {base_dir}.")
        files = {c: sorted(os.listdir(base_dir / c)) for c in conditions}
        subjects = {c: {subject_id(f) for f in files[c]} for c in conditions}
        common = set.intersection(*subjects.values())
        for condition in conditions:
            out_dir = ctx.output(condition)
            for file in files[condition]:
                subject = subject_id(file)
                if subject in common:
                    shutil.copyfile(base_dir / condition / file, out_dir / file)
                    ctx.processed(f"{condition}/{file}")
                else:
                    missing = [c for c in conditions if subject not in subjects[c]]
                    ctx.reject(base_dir / condition / file, f"{condition}/{ALIGNING_DIR}",
                               f"subject {subject} is missing from {', '.join(missing)}")
        ctx.log.info(f"{len(common)} subjects in all of {', '.join(conditions)}")
