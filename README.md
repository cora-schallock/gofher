# gofher
gofher - Galactic Orientation From Histogram Estimated Reddening

From: Schalllock et al. 2024

Contact: coraschallock[at symbol]gmail[dot]com

## Summary:
Figure out the orientation of spiral galaxies by locating which side of a galaxy's minor axis is redder

## Steps to recreate gofher-data:
1) Download the [spin-parity-catalog-data repo](https://github.com/cora-schallock/spin-parity-catalog-data)
2) Activate gofher_env
3) Navigate to example/run_gofher_on_catalog.ipynb
4) Update the `path_to_catalog_data` and `path_to_output` varaibles. `path_to_catalog_data` refers to the path of the downloaded spin-parity-catalog-data.
5) Run the notebook

## Bug reporting:
Any bugs found should be reported via [the issues tab](https://github.com/cora-schallock/gofher/issues).

## License:
Gofher is issued under the [GNU GPL 3.0 License](https://github.com/cora-schallock/gofher/blob/main/LICENSE) and as such the code may be adapted, so long as the changes are made avalible, the original source is listed, and any updates/forks/etc. of the codebase use the same license. In addition we request that anyone who publishes a paper using gofher's codebase or a modified version of gofher's codebase cite this paper.
