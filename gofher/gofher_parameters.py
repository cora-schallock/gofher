from mask import create_bisection_mask, create_ellipse_mask

class gofher_parameters:
    """Contains gofher ellipse parameters"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0
        self.b = 0.0
        self.theta = 0.0

    def load_from_sep_object(self,sep_object):
        self.x = sep_object['x']
        self.y = sep_object['y']
        self.a = sep_object['a']
        self.b = sep_object['b']
        self.theta = sep_object['theta']

    def is_valid(self):
        return self.x > 0.0 and self.y > 0.0 and self.a > 0.0 and self.b > 0.0

    def create_ellipse_mask(self,shape,r=1.0):
        """using gofher_params specifying an ellipse create an ellipse mask"""
        if not self.is_valid: 
            raise ValueError("Gofher Parameters (x={},y={},a={},b={}) must be positive values to create an ellipse mask".format(self.x,self.y,self.a,self.b))
        if not isinstance(shape, tuple) or not len(shape) == 2 or not isinstance(shape[0],int) or not isinstance(shape[1],int) or min(shape) <= 0:
            raise ValueError("shape ({}) must be a tuple of 2 positive integers to create an ellipse mask ".format(shape))
        return create_ellipse_mask(self.x,self.y,self.a,self.b,self.theta,r,shape)

    def create_bisection_mask(self,shape):
        """using gofher_params create a bisection mask"""
        if not self.is_valid: 
            raise ValueError("Gofher Parameters (x={},y={},a={},b={}) must be positive values to create a bisection mask".format(self.x,self.y,self.a,self.b))
        if not isinstance(shape, tuple) or not len(shape) == 2 or not isinstance(shape[0],int) or not isinstance(shape[1],int) or min(shape) <= 0:
            raise ValueError("shape ({}) must be a tuple of 2 positive integers to create an ellipse mask ".format(shape)) 
        return create_bisection_mask(self.x,self.y,self.theta,shape)

    