# -*- coding: utf-8 -*-
"""
The scripts that used to hard-code MPI cluster paths take their inputs as arguments and fail
with a message that names the expected file and its download page.
"""
import pytest

from human_body_prior.evaluations import run_on_amass
from human_body_prior.tools import bodypart2vertexid


def test_smplx_part_ids_requires_the_model_paths():
    with pytest.raises(ValueError, match='smpl-x.is.tue.mpg.de'):
        bodypart2vertexid.smplx_part_ids()


def test_smplx_part_ids_reports_a_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='smpl-x.is.tue.mpg.de'):
        bodypart2vertexid.smplx_part_ids(bm_fname=str(tmp_path / 'model.npz'), joints_fname=str(tmp_path / 'joints.npz'))


def test_smplh_part_ids_requires_the_model_path():
    with pytest.raises(ValueError, match='mano.is.tue.mpg.de'):
        bodypart2vertexid.smplh_part_ids()


def test_smplh_part_ids_reports_a_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='mano.is.tue.mpg.de'):
        bodypart2vertexid.smplh_part_ids(bm_fname=str(tmp_path / 'model.npz'))


def test_bodypart2vertexid_cli_needs_the_model_file(capsys):
    with pytest.raises(SystemExit):
        bodypart2vertexid.main(['smplh'])
    assert '--bm-fname' in capsys.readouterr().err


def test_run_on_amass_requires_the_experiment_dir():
    with pytest.raises(ValueError, match='smpl-x.is.tue.mpg.de'):
        run_on_amass.main()


def test_run_on_amass_reports_a_missing_experiment_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match='smpl-x.is.tue.mpg.de'):
        run_on_amass.main(expr_dir=str(tmp_path / 'missing'))
