import sep
import numpy as np

def calc_sep_size(a: float, b: float)-> float:
    """Calculates area of ellipse
    
    Args:
        a: semi-major axis length
        b: semi-minor axis length
        
    Returns:
        Area of ellipse
    """
    return 3.14*a*b

def calc_dist(cm_x: float,cm_y: float, el_x: float,el_y: float):
    """Run source extractor on a given galaxy
    
    Args:
        cm_x: approximate center x cordinate of primary object of interest (used in heuristic)
        cm_y: approximate center y cordinate of primary object of interest (used in heuristic)
        el_x: x cordinate of current object
        el_y: y cordinate of current object
        
    Returns:
        distance of (el_x,el_y) from (cm_x,cm_y) using L2 norm
    """
    return (((cm_x-el_x)*(cm_x-el_x)) + ((cm_y-el_y)*(cm_y-el_y)))**0.5

def run_sep(data, cm_x, cm_y):
    """Run source extractor on a given galaxy
    
    Args:
        data: data to run source extraction on
        cm_x: approximate center x cordinate of primary object of interest (used in heuristic)
        cm_y: approximate center y cordinate of primary object of interest (used in heuristic)"""
    data = data.byteswap().newbyteorder()
    bkg = sep.Background(data)
    data_sub = data - bkg

    mu_bkg = bkg.globalback
    
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
    
    the_el_sep = None
    the_el_sep_dist = np.inf

    for i in objects:
        if calc_sep_size(i['a'],i['b']) < 0.001*data.shape[0]*data.shape[1]: continue
        
        d = calc_dist(cm_x,cm_y,i['x'],i['y'])
        if d < the_el_sep_dist:
            the_el_sep = i; the_el_sep_dist = d
    
    return the_el_sep, mu_bkg