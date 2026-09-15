import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colors as colors
import matplotlib.patches as patches

from astropy.visualization import make_lupton_rgb

import numpy as np

from gofher.gofher_parameters import GofherParameters
from gofher.galaxy_band_pair import GalaxyBandPair
from gofher.utils import is_float_int, is_2d_bool_array, is_2d_same_shape_arrays, is_int, is_2d_array

BLACK_GRADIENT_PIXEL = np.array([0, 0, 0, 255], dtype=np.uint8)  # Opaque black
POS_MASK_PIXEL = np.array([190, 67, 159, 255], dtype=np.uint8) #BE439F
NEG_MASK_PIXEL = np.array([222, 158, 54, 255], dtype=np.uint8) #BE439F
BOUNDS_PIXEL = np.array([255, 0, 0, 255], dtype=np.uint8) # red
CLEAR_GRADIENT_PIXEL = np.array([0, 0, 0, 0], dtype=np.uint8)   # Transparent clear

POS_HIST_COLOR = '#BE439F'
NEG_HIST_COLOR = '#DE9E36'
HIST_BIN_ALPHA = 0.35

def plot_mask(ax: matplotlib.axes._axes.Axes,
              ref_band_data: np.ndarray,
              gp: GofherParameters,
              area_to_norm: np.ndarray,
              pixel_bounds: tuple[float] | None = None,
              c: float = 4):
    """Plot the masks used by gofher
    
    Graph displays the following:
        pos_side (i.e. pos bisection mask and area_to_norm) - semi-transparent POS_MASK_PIXEL
        neg_side (i.e. neg bisection mask and area_to_norm) - semi-transparent NEG_MASK_PIXEL
        cropping box - if pixel bounds are passed, displays the cropping bounds used to display the diff images
        arrows - indicating the positive and negative side
        background - the data from the ref_band fits

    Args:
        ax: the matplot axes the graph will be added to
        ref_band_data: the fits data of the ref_band
        gp: the gofher parameters used
        area_to_norm: a boolean np.ndarray indicating pixels used in gofher
        pixel_bounds: [xmin,xmax,ymin,ymax] that specify the x and y ranges of the
            diff image croppings displayed for each waveband pair
        c: the number of std (i.e. mean +/- c * std) to use for vmin/vmax 
            for plotting ref_band_data
    """
    # Validate input:
    if not isinstance(ax,matplotlib.axes._axes.Axes):
        raise TypeError("ax must be matplotlib axes")
    
    if not is_2d_array(ref_band_data):
        raise TypeError("ref_band_data must be 2D np.ndarray")

    if not isinstance(gp,GofherParameters):
        raise TypeError("gp must be GofherParameters")

    if not is_2d_bool_array(area_to_norm):
        raise TypeError("area_to_norm must be 2D boolean np.ndarray")

    if not is_2d_same_shape_arrays(ref_band_data, area_to_norm):
        raise ValueError("ref_band_data and area_to_norm must have same shape")

    if pixel_bounds is not None:
        if not isinstance(pixel_bounds,(tuple,list)):
            raise TypeError("pixel_bounds must be tuple/list of ints")
        
        if len(pixel_bounds) != 4:
            raise ValueError("pixel_bounds must have exactly 4 elements")

        for each_bounds in pixel_bounds:
            if not is_int(each_bounds):
                raise TypeError("each element in pixel_bounds must be int")

            if each_bounds < 0:
                raise ValueError("each element in pixel_bounds must be strictly positive")

        if pixel_bounds[0] >= pixel_bounds[1]:
            raise ValueError("pixel_bounds[0] must be larger than pixel_bounds[1]")
        
        if pixel_bounds[2] >= pixel_bounds[3]:
            raise ValueError("pixel_bounds[2] must be larger than pixel_bounds[3]")

    if not is_float_int(c):
        raise TypeError("c must be int")

    if c <= 0:
        raise ValueError("c must be strictly positive")
    
    # Display the ref_band_data:
    vmin = np.mean(ref_band_data) - c*np.std(ref_band_data)
    vmax = np.mean(ref_band_data) + c*np.std(ref_band_data)
    ax.imshow(ref_band_data,origin='lower',cmap='grey',vmin=vmin, vmax=vmax)

    # Get the pos and neg labels:
    pos_label, neg_label = gp.get_pos_neg_labels()

    # If pixel bounds are passed, display cropping of diff images:
    if pixel_bounds is not None:
        xmin = pixel_bounds[0]
        xmax = pixel_bounds[1]
        ymin = pixel_bounds[2]
        ymax = pixel_bounds[3]

        width = xmax-xmin
        height = ymax-ymin
        
        rect = patches.Rectangle((pixel_bounds[0],
                                  pixel_bounds[2]),
                                  width,
                                  height,
                                  linewidth=0.5,
                                  edgecolor=BOUNDS_PIXEL/255,
                                  fill=False)
        ax.add_patch(rect)

    # Calculate where to place arrows and labels:
    # Programmer Note: the values I used were mostly my aesthetic judgement to make the plot
    #   both readable and pretty. I will explain my logic below but if the values seem arbitrary
    #   they more or less are lol. So feel free to adjust, I just ask you think of edge values so
    #   that they too are readable and somewhat pretty.
    #
    # Logic behind the values I used:
    #   Arrow:
    #       * points to the minor axis at a location that is approximately 
    #           half way between origin/ side of minor axis
    #       * if the arrows location is too close to nearest edge of image (or past it)
    #           defaults to half the distance between origin and closest edge minus the margin
    #   Label:
    #       * located at small offset angle from minor axis
    #       * location will be 2*minor axis length from origin minus a small side margin
    #       * if the label location is too close to nearest edge of image (or past it)
    #           defaults to two times the distance between origin and closest 
    #           edge minus the margin  
    side_margin = 10
    arrow_theta = np.pi/2 + gp.theta
    label_theta = 2*np.pi/3 + gp.theta

    distance_to_edges = [gp.h,
                         gp.shape[1]-gp.h,
                         gp.k,
                         gp.shape[0]-gp.k]

    max_radius = np.min(distance_to_edges)-side_margin

    arrow_radius = min(gp.b*0.5,max_radius)
    label_radius = min(gp.a*2.0,max_radius)

    pos_arrow_xy = (gp.h + arrow_radius * np.cos(arrow_theta),
                    gp.k + arrow_radius * np.sin(arrow_theta))
    pos_label_xy = (gp.h + label_radius * np.cos(label_theta),
                    gp.k + label_radius * np.sin(label_theta))
    neg_arrow_xy = (gp.h - arrow_radius * np.cos(arrow_theta),
                    gp.k - arrow_radius * np.sin(arrow_theta))
    neg_label_xy = (gp.h - label_radius * np.cos(label_theta),
                    gp.k - label_radius * np.sin(label_theta))

    # Add arrow and label:
    ax.annotate(
        f'{pos_label} side',           # The text label
        xy=pos_arrow_xy,               # (x, y) coordinate where the arrow points
        xytext=pos_label_xy,            # (x, y) coordinate where the text sits
        arrowprops=dict(              # Dictionary defining arrow style
            facecolor='black',        # Color of the arrow body
            arrowstyle='->',          # Arrow style type (simple line and pointer)
            connectionstyle='arc3'    # Curve style (straight line)
        ),
        bbox=dict(
            boxstyle='round,pad=0.3',    # Shapes: 'square', 'round', 'sawtooth'
            facecolor=POS_MASK_PIXEL/255,         # Solid background color
            edgecolor='black',          # Border line color
            alpha=1.0                   # 1.0 is completely solid; 0.5 is semi-transparent
        )
    )

    ax.annotate(
            f'{neg_label} side',           # The text label
            xy=neg_arrow_xy,               # (x, y) coordinate where the arrow points
            xytext=neg_label_xy,            # (x, y) coordinate where the text sits
            arrowprops=dict(              # Dictionary defining arrow style
                facecolor='black',        # Color of the arrow body
                arrowstyle='->',          # Arrow style type (simple line and pointer)
                connectionstyle='arc3'    # Curve style (straight line)
            ),
            bbox=dict(
                boxstyle='round,pad=0.3',    # Shapes: 'square', 'round', 'sawtooth'
                facecolor=NEG_MASK_PIXEL/255,         # Solid background color
                edgecolor='black',          # Border line color
                alpha=1.0                   # 1.0 is completely solid; 0.5 is semi-transparent
            )
    )
    
    # Display the pos and neg bisection masks combined with the area to norm
    rows, cols = gp.shape
    pos, neg = gp.create_bisection_masks()
    bisection_mask = np.zeros((rows,cols,4), dtype=np.uint8)
    bisection_mask[np.logical_and(pos,area_to_norm)] = POS_MASK_PIXEL
    bisection_mask[np.logical_and(neg,area_to_norm)] = NEG_MASK_PIXEL
    ax.imshow(bisection_mask,alpha=0.25,origin='lower')

    # Title the plot:
    ax.set_title(f"{gp.name} masks")


def plot_diff_histogram(ax: matplotlib.axes._axes.Axes, 
                        bp: GalaxyBandPair, 
                        range: tuple[float, float] | None = None):
    """Plot a histogram of the diff values split between the pos and neg sides
    
    Args:"""

    # Validated inputs:
    if not isinstance(ax,matplotlib.axes.Axes):
        raise TypeError("ax must be matplotlib.axes.Axes")

    if not isinstance(bp,GalaxyBandPair):
        raise TypeError("bp must be GalaxyBandPair")

    if range is not None and not isinstance(range,(tuple, list)):
        raise TypeError("range must be tuple or list if provided or None otherwise")

    if range is not None and len(range) != 2:
        raise ValueError("range must be 2 element tuple or list if provided or None otherwise")

    if not is_float_int(range[0]) or not is_float_int(range[1]):
        raise TypeError("range elements must be either floats or ints")

    # Calculate number of bins:
    number_of_bins = int(np.sqrt(len(bp.pos_side_values) + len(bp.neg_side_values)))

    # Plot the histogram:
    ax.hist(bp.pos_side_values,
            range=range,
            color=POS_HIST_COLOR,
            alpha = HIST_BIN_ALPHA, 
            density=True,
            bins=number_of_bins)
    
    ax.hist(bp.neg_side_values,
            range=range,
            color=NEG_HIST_COLOR, 
            alpha = HIST_BIN_ALPHA, 
            density=True, 
            bins=number_of_bins)

    # Plot the mean of the pos and neg side:
    ax.axvline(bp.pos_side_mean,
               color=POS_HIST_COLOR,
               linestyle="-.",
               gapcolor="white",
               label=f"{bp.pos_side_label} μ = {bp.pos_side_mean:.4f}")
    ax.axvline(bp.neg_side_mean,
               color=NEG_HIST_COLOR,
               linestyle="--",
               gapcolor="white",
               label=f"{bp.neg_side_label} μ = {bp.neg_side_mean:.4f}")

    # Add labels and title:
    ax.set_title(f"{str(bp)}: {bp.redder_side_label}")
    ax.set_xlabel("Diff Value")
    ax.set_ylabel("Density")
    ax.legend()

def plot_diff_image(ax: matplotlib.axes._axes.Axes, 
                    bp: GalaxyBandPair,
                    area_to_norm: np.ndarray,
                    pixel_bounds: tuple[int] | None = None):
    """Plot the diff image
    
    Args:
        ax: the matplot lib axes to plot on
        bp: the band pair to use
        area_to_norm: the 2D boolean np.ndarray that specifies which pixels
            are used by gofher
        pixel_bounds: [xmin,xmax,ymin,ymax] specify the cropping bounds of diff
            image if provided, otherwise uses no cropping

    Returns:
        (imshow, ticks)
    """

    # Validate the input:
    if not isinstance(ax,matplotlib.axes.Axes):
        raise TypeError("ax must be matplotlib.axes.Axes")
    
    if not isinstance(bp,GalaxyBandPair):
        raise TypeError("bp must be GalaxyBandPair")

    if not is_2d_bool_array(area_to_norm):
        raise TypeError("area_to_norm must be 2D boolean numpy array")

    if not is_2d_same_shape_arrays(bp.get_diff_image(),area_to_norm):
        raise ValueError("diff_image ande area_to_norm must both be same size numpy array")

    if pixel_bounds is not None and not isinstance(pixel_bounds,(list,tuple)):
        raise TypeError("pixel_bounds must be tuple or list of ints if provide; None otherwise")

    if pixel_bounds is not None and len(pixel_bounds) != 4:
        raise ValueError("pixel_bounds must have 4 elements if provided")

    if pixel_bounds is not None:
        for each_bound in pixel_bounds:
            if not is_int(each_bound):
                raise TypeError("all elements of pixel_bounds must be int")

    if pixel_bounds is not None and pixel_bounds[0] > pixel_bounds[2]:
        raise ValueError("pixel_bounds xmin (at index 0) must be smaller than xmax (at index 2)")

    if pixel_bounds is not None and pixel_bounds[1] > pixel_bounds[3]:
        raise ValueError("pixel_bounds ymin (at index 1) must be smaller than ymax (at index 3)")

    # Mask out all pixels not included in area_to_norm:
    outside_mask = np.where(area_to_norm[..., None], CLEAR_GRADIENT_PIXEL, BLACK_GRADIENT_PIXEL)

    # Get the diff image and find the range of values:
    diff_image = bp.get_diff_image()
    the_min = np.min(diff_image[area_to_norm])
    the_max = np.max(diff_image[area_to_norm])

    # Create a normalization for the colors displayed:
    #   0.0 is in the middle
    #   Since we are using a RdBu color map, redder regions will appear red
    #   bluer regions will appear blue
    norm = colors.TwoSlopeNorm(vcenter=0, vmin=the_min, vmax=the_max)

    # Use tick marks on colorbar with max/min value and 0 value:
    ticks = np.array(sorted([the_max,0.0,the_min]))

    # Plot the diff image and return it to add colorbar later:
    diff = ax.imshow(diff_image,origin='lower',cmap='RdBu',norm=norm)
    ax.imshow(outside_mask,origin='lower')

    # Set title:
    ax.set_title(f"{str(bp)} diff")

    # If pixel_bounds is provided, crop image to include region in pixel bounds:
    if pixel_bounds is not None:
        ax.set_xlim(pixel_bounds[0],pixel_bounds[1])
        ax.set_ylim(pixel_bounds[2],pixel_bounds[3])

    return diff, ticks
