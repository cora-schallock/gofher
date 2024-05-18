def pos_neg_label_from_theta(theta):
    """get pos/neg label from theta"""
    new_theta = theta % 360
    pos = '-'
    neg = '-'
    if new_theta <= 22.5:
        pos = 'n'
        neg = 's'
    elif new_theta > 22.5 and new_theta < 67.5:
        pos = 'ne'
        neg = 'sw'
    elif new_theta >= 67.5 and new_theta <= 112.5:
        pos = 'e'
        neg = 'w'
    elif new_theta > 112.5 and new_theta < 157.5:
        pos = 'se'
        neg = 'nw'
    elif new_theta >= 157.5 and new_theta <= 202.5:
        pos = 's'
        neg = 'n'
    elif new_theta > 202.5 and new_theta < 247.5:
        pos = 'sw'
        neg = 'ne'
    elif new_theta >= 247.5 and new_theta <= 292.5:
        pos = 'w'
        neg = 'e'
    elif new_theta > 292.5 and new_theta < 337.5:
        pos = 'nw'
        neg = 'se'
    else:
        pos = 'n'
        neg = 's'
    return pos.upper(),neg.upper()

def get_opposite_label(label):
    opposite_dict = {"n":"s","nw":"se","w":"e","sw":"ne","s":"n","se":"nw","e":"w","ne":"s"}
    return opposite_dict.get(label.strip().lower(),"")