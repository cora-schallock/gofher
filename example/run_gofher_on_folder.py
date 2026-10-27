from pathlib import Path
import logging

#fix for module missing error:
import sys
import os
sys.path.append(os.path.abspath(".."))

from gofher.sparcfire import gofher_params_from_sparcfire_csv
from gofher.file_helper import assure_folder_exists
from gofher.galaxy import Galaxy

# Directories and output location:
FITS_DIR = "/extra/wayne2/research/drdavis/SDSS/FITS/2015/color"
SPARCFIRE_DIR = "/extra/wayne2/research/drdavis/SDSS/SpArcFiRe/2016-09"
OUTPUT_DIR = "/extra/wayne2/preserve/cschallo/gofher/2026_10_run"
LOG_DIR = "/extra/wayne2/preserve/cschallo/gofher/2026_10_run/logs"

# The waveband to use for SpArcFiRe fit:
REF_BAND = "r"

# GOFHER specific parameters:
BLUER_TO_REDDER_BANDS = ['g','r','i','z']
SPARCFIRE_BULGE_DISK_F = 0.25

def _get_fits_path(folder, name, band):
    return Path(FITS_DIR) / band / folder / f"{name}.fits"

def _get_sparcfire_csv_path(folder, ref_band):
    return Path(SPARCFIRE_DIR) / ref_band / folder / "galaxy.csv"

def _get_output_dir(folder, name):
    return Path(OUTPUT_DIR) / folder / name

def run_on_galaxy(gofher_param, folder):
    """Run GOFHER on single galaxy"""
    # Get galaxy name:
    name = gofher_param.name

    # Create a Galaxy and add all wavebands to it:
    the_galaxy = Galaxy(gofher_param)

    for band in BLUER_TO_REDDER_BANDS:
        fits_path = _get_fits_path(folder, gofher_param.name, band)
        the_galaxy.construct_galaxy_band_from_fits(band,fits_path)

    # Run GOFHER on the galaxy:
    the_galaxy.run(bluer_to_redder_bands=BLUER_TO_REDDER_BANDS,
                sparcfire_bulge_disk_f=SPARCFIRE_BULGE_DISK_F,
                area_to_consider=None,
                fail_silently_on_missing_band=False)

    # Create an output folder:
    output_folder = _get_output_dir(folder, name)
    assure_folder_exists(output_folder)

    # Save a figure visualizing GOFHER's output:
    the_galaxy.plot_figure(output_folder / f"{name}.png")

    # Save a csv with GOFHER's data:
    the_galaxy.output_to_csv(output_folder / f"{name}.csv")

    # Save the normalization:
    the_galaxy.save_normalizations(output_folder)

    #save the pos/neg side values:
    the_galaxy.save_pos_and_neg_side_values(output_folder)

def run_on_folder(folder):
    """Run GOFHER on entire folder"""
    logging.info(f"Begin run on folder {folder}")

    sparcfire_csv_path = _get_sparcfire_csv_path(folder, REF_BAND)
    logging.info(f"Reading in SpArcFiRe csv: {sparcfire_csv_path}")
    
    gals = gofher_params_from_sparcfire_csv(folder, REF_BAND)
    logging.info(f"Read in {len(gals)} galaxies")

    i = 1
    finished_count = 0
    failed_count = 0
    for each_param in gals:
        try:
            name = each_param.name
            logging.info(f"Running on galaxy #{i}: {name}")
            run_on_galaxy(each_param, folder)
            finished_count += 1
            logging.info(f"Finished on galaxy #{i}: {name}")

        except (TypeError, 
                ValueError, 
                RuntimeError,
                KeyError,
                ZeroDivisionError,
                FileNotFoundError) as err:
            logging.error(f"error galaxy {name}: {err}")
            failed_count += 1
            continue
        except Exception as err:
            logging.error(f"error galaxy {name}: {err}")
            raise 
        i += 1

    logging.info(f"End run on folder {folder}")
    logging.info(f"{finished_count} finished {failed_count} failed")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_to_run_on = sys.argv[1]
    else:
        raise RuntimeError("missing 1 required param: folder")

    # Configure logging globally
    logging.basicConfig(
        level=logging.INFO,                           # Track INFO and higher (ignores DEBUG)
        format="%(asctime)s - %(levelname)s - %(message)s", # Customize the look
        filename=Path(LOG_DIR) / f"{folder_to_run_on}.log",                           # Save logs to this file (removes console output)
        filemode="w"                                  # 'w' to overwrite each run, 'a' to append
    )

    run_on_folder(folder_to_run_on)

