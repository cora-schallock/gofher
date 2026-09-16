"""Test all functions in gofher/galaxy.py

The script is run using the commend: python -m pytest
"""
from pathlib import Path
import numpy as np
import uuid

import pytest

from gofher.file_helper import read_array_file
from gofher.galaxy import Galaxy
from gofher.gofher_parameters import GofherParameters
from gofher.sparcfire import gofher_params_from_sparcfire_csv
from gofher.utils import is_float_int_array, is_2d_bool_array

RESIDUAL_TOLERANCE = 1e-6

#IMPORTANT: Make sure these folder/file names reflect test file structure:
TEST_GALAXY_DIR = "NGC2347_SDSS_psf4_background_256"
TEST_SPARCFIRE_DIR = "sparcfire_r_band_output"
TEST_FITS_DIR = "fits"
TEST_CSV_FILE_NAME = "NGC2347_r.csv"

def _get_test_csv_path() -> str:
    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_SPARCFIRE_DIR, TEST_CSV_FILE_NAME))

def _get_test_fits_path(band) -> str:
    """Helper function to get path to test fits file"""
    fits_file_name = f"NGC2347_{band}.fits"

    p = Path(__file__).resolve().parent
    return str(p.joinpath("data", TEST_GALAXY_DIR, TEST_FITS_DIR, fits_file_name))

def _get_test_galaxy_param() -> GofherParameters:
    # Read in the gofher params from the provided SpArcFiRe CSV:
    gofher_params = gofher_params_from_sparcfire_csv(_get_test_csv_path())
    
    # For setting up this test, we need to verify the gofher_params 
    # is a list containning one gofher. If there is an issue with this,
    # double check gofher_params_from_sparcfire_csv()
    # Note this should already be checked in test_sparcifre.py 
    assert isinstance(gofher_params,list)
    assert len(gofher_params) == 1
    assert isinstance(gofher_params[0], GofherParameters)

    return gofher_params[0]

def _build_test_galaxy(bands: list[str] = ["g", "r"],
                       run_with_default_args: bool = True) -> Galaxy:
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()

    # Construct the galaxy and bands:
    the_galaxy = Galaxy(gofher_param)

    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

    if run_with_default_args:
        the_galaxy.run(bands)
    
    return the_galaxy


@pytest.mark.parametrize("gofher_params, expected_exception", [
    ("", TypeError)
])
def test_galaxy_initalization_excpetions(gofher_params, expected_exception):
    """Test exceptions expected from galaxy __init__"""
    with pytest.raises(expected_exception):
        Galaxy(gofher_params)

def test_galaxy_initalization():
    """Test galaxy initalziation
    
    Note to programmer - This test may seem silly, but it is verifying that 
    this doesn't cause an exception. It also is a way to validate the test 
    SpArcFiRe csv is being read correctly
    """

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()

    # Construct the galaxy:
    Galaxy(gofher_param)

@pytest.mark.parametrize("band, expected_exception", [
    ([], TypeError)
])
def test_has_band_exception(band, expected_exception):
    """Test exceptions raised by Galaxy.has_band()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Check exception is raised:
    with pytest.raises(expected_exception):
        the_galaxy.has_band(band)

def test_has_band():
    """Test Galaxy.has_band()"""

    # Specify test band and fits:
    test_band = "g"
    test_fits = _get_test_fits_path(test_band)

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Verify the galaxy doesn't currently have band:
    assert not the_galaxy.has_band(test_band)

    # Add the test band to the galaxy:
    the_galaxy.construct_galaxy_band_from_fits(test_band, test_fits)

    # Verify the galaxy now has band:
    assert the_galaxy.has_band(test_band)

@pytest.mark.parametrize("band, expected_exception", [
    ([], TypeError),
    ("missingkey", KeyError)
])
def test_get_band_exception(band, expected_exception):
    """Test exceptions raised by Galaxy.get_band()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Check exception is raised:
    with pytest.raises(expected_exception):
        the_galaxy.get_band(band)

def test_get_band():
    """Test Galaxy.get_band()"""

    # Specify test band and fits:
    test_band = "g"
    test_fits = _get_test_fits_path(test_band)

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Add the test band to the galaxy:
    the_band = the_galaxy.construct_galaxy_band_from_fits(test_band, test_fits)

    assert the_galaxy.get_band(test_band) == the_band

@pytest.mark.parametrize("band, fits_path, expected_exception", [
    ([],_get_test_fits_path("g"), TypeError),
    ("",_get_test_fits_path("g"), ValueError),
    ("g-",_get_test_fits_path("g"), ValueError),
    ("g_",_get_test_fits_path("g"), ValueError),
    ("g",{}, TypeError),
    ("g","not_a_fits.pdf", ValueError),
    ("g","fake_file_path.fits", ValueError)
])
def test_construct_galaxy_band_from_fits_excpetions(band,fits_path,expected_exception):
    """Test exceptions from construct_galaxy_band_from_fits()"""

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()

    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # Validate the correct excpetion is raised:
    with pytest.raises(expected_exception):
        the_galaxy.construct_galaxy_band_from_fits(band,fits_path)

def test_construct_galaxy_band_from_fits():
    """Test galaxy.construct_galaxy_band_from_fits()"""
    
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
    
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)
    bands = ['g','r','i','z','u']
    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

@pytest.mark.parametrize("bands, f, area, fail_silently, expected_exception", [
    ([], 1.0, True, None, ValueError),
    (["a",{}], 1.0, None, True, TypeError),
    (["g","r"], dict(), None, True, TypeError),
    (["g","r"], -1.0, None, True, ValueError),
    (["g","r"], 1.5, None, True, ValueError),
    (["g","r"], 0.5, "", True, TypeError),
    (["g","r"], 0.5, np.ones((10,10),bool), True, ValueError),
    (["g","r"], 0.5, None, "", TypeError)
])
def test_run_exceptions(bands, f, area, fail_silently, expected_exception):
    """Verify the exceptions raised in Galaxy.run()"""
    # Get test galaxy with bands but DON'T RUN:
    the_galaxy = _build_test_galaxy(run_with_default_args=False)

    # Validate the correct excpetion is raised:
    with pytest.raises(expected_exception):
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=area,
                       fail_silently_on_missing_band=fail_silently)

def test_run_area_to_consider_exception():
    """Verify the exceptions raised in Galaxy.run() if area_to_consider has no Pixels"""
    # Get galaxy with band but DON'T RUN:
    the_galaxy = _build_test_galaxy(run_with_default_args=False)

    # Get shape of fits
    shape = the_galaxy.gofher_params.shape

    # If area_to_consider has all Falses, it raises a ValueError exception:
    no_area = np.zeros(shape,bool)
    with pytest.raises(ValueError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=0.5,
                       area_to_consider=no_area,
                       fail_silently_on_missing_band=False)

    # If area_to_consider is passed in with at least one True
    #   but shares no intersecting True pixels ellipse mask/ valid mask
    #   it reaises a RunTime exception:
    no_area[0,0] = True
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=0.5,
                       area_to_consider=no_area,
                       fail_silently_on_missing_band=False)

    # Now we will in all True to area_to_consider and it should work:
    area = np.ones(shape,bool)
    the_galaxy.run(bluer_to_redder_bands=["g","r"],
                   sparcfire_bulge_disk_f=0.5,
                   area_to_consider=area,
                   fail_silently_on_missing_band=False)

def test_run_no_pos_neg_area_exception():
    """Verify an exception is raised in pos mask or neg mask has no area"""

    # Specify the sparcfire bulge disk f to use:
    sparcfire_bulge_disk = 0.5

    # Get test galaxy with bands but DON'T RUN:
    the_galaxy = _build_test_galaxy(run_with_default_args=False)
    gofher_param = the_galaxy.gofher_params

    gofher_param.calculate_from_sparcfire(sparcfire_bulge_disk)
    pos_mask, neg_mask = gofher_param.create_bisection_masks()

    # This should work:
    the_mask = np.logical_or(pos_mask, neg_mask)
    the_galaxy.run(bluer_to_redder_bands=["g","r"],
                   sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                   area_to_consider=the_mask)

    # When area_to_consider shares no intersecting True pixels with pos side
    #   (i.e. in this toy example we use neg_mask) it raises a Runtime Exception
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                       area_to_consider=neg_mask)

    # When area_to_consider shares no intersecting True pixels with neg side
    #   (i.e. in this toy example we use pos_mask) it raises a Runtime Exception
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=["g","r"],
                       sparcfire_bulge_disk_f=sparcfire_bulge_disk,
                       area_to_consider=pos_mask)

def test_run_silent_fail():
    """Test fail_silently_on_missing_band parameter in Galaxy.run()"""
    # The parameters for the test:
    bands = ["g","r"]
    f = 1.0

    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy:
    the_galaxy = Galaxy(gofher_param)

    # No bands are present yet, so it causes a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=None,
                       fail_silently_on_missing_band=False)

    # Add the g band:
    the_galaxy.construct_galaxy_band_from_fits("g",_get_test_fits_path("g"))

    # r band still missing, so it casues a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.run(bluer_to_redder_bands=bands,
                       sparcfire_bulge_disk_f=f,
                       area_to_consider=None,
                       fail_silently_on_missing_band=False)

    # Add the r band:
    the_galaxy.construct_galaxy_band_from_fits("r",_get_test_fits_path("r"))

    # Now run should work without casuing an exception
    the_galaxy.run(bluer_to_redder_bands=bands,
                   sparcfire_bulge_disk_f=f,
                   area_to_consider=None,
                   fail_silently_on_missing_band=False)

def test_run():
    """Test Galaxy.run()"""
    # The parameters for the test:
    bands = ['u','g','r','i','z']
    f = 0.5
    
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
        
    # Construct the galaxy and bands:
    the_galaxy = Galaxy(gofher_param)
    for band in bands:
        the_galaxy.construct_galaxy_band_from_fits(band,_get_test_fits_path(band))

    # Run
    the_galaxy.run(bluer_to_redder_bands=bands,
                   sparcfire_bulge_disk_f=f)

@pytest.mark.parametrize("red, green, blue, expected_exception", [
    ([],"r","g", TypeError),
    ("i",[],"g", TypeError),
    ("i","r",[], TypeError),
    ("imissing","rmissing","gmissing", RuntimeError),
])
def test_make_lupton_rgb_exception(red, green, blue, expected_exception):
    # Get test gofher param:
    gofher_param = _get_test_galaxy_param()
            
    # Construct the galaxy and bands:
    the_galaxy = Galaxy(gofher_param)

    with pytest.raises(expected_exception):
        the_galaxy.make_lupton_rgb(red, green, blue)

@pytest.mark.parametrize("save_path, dpi, expected_exception", [
    ([],300, TypeError),
    ("example_output.png", [], TypeError),
    (Path("example_output.png"), -10, ValueError)
])
def test_plot_figure_exception(save_path, dpi, expected_exception):
    # Get test galaxy with bands and run:
    the_galaxy = _build_test_galaxy()

    with pytest.raises(expected_exception):
        the_galaxy.plot_figure(save_path, dpi)

def test_plot_figure_runtime_exception():
    # Specify a test file path for plots:
    random_file_path = Path(TMP_DIR / f"{uuid.uuid4()}.png")
                    
    # Construct the galaxy and bands but DON'T RUN:
    the_galaxy = _build_test_galaxy(run_with_default_args=False)

    # Now we have not run gofher so plotting should raise a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.plot_figure(random_file_path)

    # This should work:
    the_galaxy.run(["g","r"])
    the_galaxy.plot_figure(random_file_path)
    assert random_file_path.exists()

    # Cleanup output file:
    random_file_path.unlink()

    # Now if the ref band is missing it should raise a RuntimeError:
    the_galaxy.gofher_params.ref_band = "missingband"
    with pytest.raises(RuntimeError):
        the_galaxy.plot_figure(random_file_path)

    # This should work:
    the_galaxy.gofher_params.ref_band = "g"
    the_galaxy.plot_figure(random_file_path)
    assert random_file_path.exists()

    # Cleanup output file:
    random_file_path.unlink()

    # Now if the area_to_norm is missing it should raise a RuntimeError:
    the_galaxy._area_to_norm = None
    with pytest.raises(RuntimeError):
        the_galaxy.plot_figure(random_file_path)

    # Verify no output file (since last test failed):
    assert not random_file_path.exists()

@pytest.mark.parametrize("path_to_folder, expected_exception", [
    ([], TypeError),
    (Path("file.py"), ValueError),
    (Path("missingfolder"), FileNotFoundError)
])
def test_save_normalizations_exceptions(path_to_folder, expected_exception):
    the_galaxy = _build_test_galaxy()

    with pytest.raises(expected_exception):
        the_galaxy.save_normalizations(path_to_folder)

def test_save_normalizations_no_bands_exception(tmp_path: Path):
    # Specify a test folder and create it:
    random_folder = tmp_path / f"{uuid.uuid4()}"
    random_folder.mkdir(parents=True, exist_ok=True)

    # Build the galaxy, but with no bands, also DON'T RUN:
    the_galaxy = _build_test_galaxy(bands=[], run_with_default_args=False)

    # If the galaxy has no bands, then it should raise a RuntimeError:
    with pytest.raises(RuntimeError):
        the_galaxy.save_normalizations(random_folder)
    assert len(list(random_folder.iterdir())) == 0 # Verify it didn't create files

def test_save_normalizations_missing_band_exception(tmp_path: Path):
    # Specify a test folder and create it:
    random_folder = tmp_path / f"{uuid.uuid4()}"
    random_folder.mkdir(parents=True, exist_ok=True)

    # Build the galaxy, add bands, and run:
    the_galaxy = _build_test_galaxy(bands=["g","r"])

    # Add an additional waveband (so 3 in total) and now a RuntimeError should be raised:
    the_galaxy.construct_galaxy_band_from_fits("i",_get_test_fits_path("i"))
    with pytest.raises(RuntimeError):
        the_galaxy.save_normalizations(random_folder)

    assert len(list(random_folder.iterdir())) < 4 # Verify it didn't create files: area_to_norm + 3 bands (so 4 if everything works)

def test_save_normalizations_no_area_to_norm_exception(tmp_path: Path):
    # Specify a test folder and create it:
    random_folder = tmp_path / f"{uuid.uuid4()}"
    random_folder.mkdir(parents=True, exist_ok=True)

    # Build galaxy with 2 bands and run:
    the_galaxy = _build_test_galaxy(bands=["g","r"])

    the_galaxy._area_to_norm = None
    with pytest.raises(RuntimeError):
        the_galaxy.save_normalizations(random_folder)
    assert len(list(random_folder.iterdir())) == 0 # Verify it didn't create files

def test_save_normalizations():
    # Create a random folder for tests:
    random_folder = Path(TMP_DIR / f"{uuid.uuid4()}")
    Path.mkdir(random_folder)

    # Build a galaxy with g and r bands, run it, and save the normalization:
    test_galaxy = _build_test_galaxy(["g","r"])
    test_galaxy.save_normalizations(random_folder)

    # Get paths and assure they all exist:
    area_path = Path(random_folder / "area_to_norm.npy")
    g_path = Path(random_folder / "g_normalization.npy")
    r_path = Path(random_folder / "r_normalization.npy")

    assert area_path.exists()
    assert g_path.exists()
    assert r_path.exists()

    # Read in the save_normalizations:
    area_data = read_array_file(area_path)
    g_data = read_array_file(g_path)
    r_data = read_array_file(r_path)

    # Assure they are correct shape and type of numpy array:
    assert area_data.shape == test_galaxy.gofher_params.shape
    assert is_2d_bool_array(area_data)

    assert g_data.shape == test_galaxy.gofher_params.shape
    assert is_float_int_array(g_data)

    assert r_data.shape == test_galaxy.gofher_params.shape
    assert is_float_int_array(r_data)

    # Verify the data is what is expected:
    #   For boolean mask, must be exactly the same
    #   For float/int mask, sum of residual must be less than RESIDUAL_TOLERENCE
    area_redisual = np.logical_or(area_data, test_galaxy._area_to_norm)
    assert np.sum(area_redisual) == 0

    g_residual = np.sum(np.abs(g_data - test_galaxy.get_band("g").get_normalization()))
    assert np.sum(g_residual) < RESIDUAL_TOLERANCE

    r_residual = np.sum(np.abs(r_data - test_galaxy.get_band("r").get_normalization()))
    assert np.sum(r_residual) < RESIDUAL_TOLERANCE

    random_folder.rmdir()

def test_get_csv_dict_exceptions():
    # Build a galaxy with bands but DON'T RUN it:
    test_bands = ["g","r"]
    test_galaxy = _build_test_galaxy(bands=test_bands,
                                     run_with_default_args=False)

    # Run has not been called so it should raise a RuntimeError:
    with pytest.raises(RuntimeError):
        test_galaxy.get_csv_dict()

    # To simulate missing band pairs, but has vote count we will manually set it:
    test_galaxy.pos_label = "pos"
    test_galaxy.vote_count_pos = 1
    test_galaxy.pos_label = "neg"
    test_galaxy.vote_count_pos = 0

    # Since there are no band pairs, it will raise a RuntimeError:
    with pytest.raises(RuntimeError):
        test_galaxy.get_csv_dict()

    # If we run it, it creates the band pairs it should work:
    test_galaxy.run(test_bands)
    test_galaxy.get_csv_dict()

    # Now we will manually change the vote counts so it should raise a RuntimeError:
    test_galaxy.vote_count_pos = 0
    test_galaxy.vote_count_neg = 0
    with pytest.raises(RuntimeError):
        test_galaxy.get_csv_dict()

    # To fix this we will just set a made up vote count, now it should work:
    test_galaxy.vote_count_pos = 1
    test_galaxy.vote_count_neg = 0
    test_galaxy.get_csv_dict()

    # Now we will clear a normalization/ classification and it should raise a RunTimeError
    test_galaxy._band_pairs[0].redder_side_label = INDETERMINANT_VOTE_LABE
    with pytest.raises(RuntimeError):
        test_galaxy.get_csv_dict()


def test_get_csv_dict():
    # Build a galaxy with bands and run it
    test_galaxy = _build_test_galaxy()
    the_dict = test_galaxy.get_csv_dict()

    assert isinstance(the_dict, dict)
    assert len(the_dict) > 0


    

    

#TODO: output_to_csv, test plot output