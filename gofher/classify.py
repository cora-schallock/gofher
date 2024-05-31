def pos_neg_label_from_theta(theta: float) -> tuple[str,str]:
    """Return positive and negative labels for bisection given theta.
    
    Args: 
        theta: angle of ellipse measured in degrees from positive X-axis of cartesian graph
            going counter-clockwise

    Returns:
        tuple[str,str]: (positive side cardinal direction label, negative side cardinal direction label)
    """
    theta = theta % 360
    if theta <= 22.5:
        return 'N', 'S'
    elif theta <= 67.5:
        return 'NE', 'SW'
    elif theta <= 112.5:
        return 'E', 'W'
    elif theta <= 157.5:
        return 'SE', 'NW'
    elif theta <= 202.5:
        return 'S', 'N'
    elif theta <= 247.5:
        return 'SW', 'NE'
    elif theta <= 292.5:
        return 'W', 'E'
    elif theta <= 337.5:
        return 'NW', 'SE'
    else:
        return 'N', 'S'

def get_opposite_label(cardinal_direction: str) -> str:
    """Return the opposite cardinal direction of the given cardinal direction.

    Args:
        cardinal_direction (str): The cardinal direction to find the opposite of.

    Returns:
        str: The opposite cardinal direction.
    """
    opposite_directions = {
        "n": "s",
        "nw": "se",
        "w": "e",
        "sw": "ne",
        "s": "n",
        "se": "nw",
        "e": "w",
        "ne": "sw"
    }
    return opposite_directions.get(cardinal_direction.strip().lower(), "")
