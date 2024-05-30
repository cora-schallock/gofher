from mask import create_bisection_mask, create_ellipse_mask

class gofher_parameters:
    """Contains gofher ellipse parameters"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0
        self.b = 0.0
        self.theta = 0.0

    def load_from_sep_object(self,sep_object: dict):
        """
        Loads the gofher ellipse parameters from a dictionary object.

        Args:
            sep_object (dict): A dictionary containing the keys 'x', 'y', 'a', 'b', and 'theta'.
            IMPORTANT: theta position angle is in radians counter clockwise from positive x axis to major axis, and lies in range [-pi/2, pi/2]
            For more information see: https://sep.readthedocs.io/en/stable/api/sep.ellipse_coeffs.html?highlight=theta

        Returns:
            None
        """
        self.y = sep_object['y']
        self.a = sep_object['a']
        self.b = sep_object['b']
        self.theta = sep_object['theta']

    def is_valid(self) -> bool:
        """
        Check if the object's attributes represent a valid ellipse.

        Returns:
            bool: True if (x,y,a,b) values are greater than 0.0, False otherwise.
        """
        return self.x > 0.0 and self.y > 0.0 and self.a > 0.0 and self.b > 0.0

    def create_ellipse_mask(self,shape,r=1.0):
        """Generate ellipse mask using the gofher_parameters
        Ellipse has center (x,y), angle theta, and semi-major/semi-major of (a,b) scaled by factor r
        IMPORTANT: theta position angle is in radians counter clockwise from positive x axis to major axis, and lies in range [-pi/2, pi/2]
        For more information see: https://sep.readthedocs.io/en/stable/api/sep.ellipse_coeffs.html?highlight=theta

        Args:
            shape: The shape of the ellipse mask
            r: how much to scale semi-major/semi-minor axis length of ellipse
                semi-major=r*a semi-minor=r*b
        Returns:
            ellipse_mask
        """
        if not self.is_valid: 
            raise ValueError("Gofher Parameters (x={},y={},a={},b={}) must be positive values to create an ellipse mask".format(self.x,self.y,self.a,self.b))
        if not isinstance(shape, tuple) or not len(shape) == 2 or not isinstance(shape[0],int) or not isinstance(shape[1],int) or min(shape) <= 0:
            raise ValueError("shape ({}) must be a tuple of 2 positive integers to create an ellipse mask ".format(shape))
        return create_ellipse_mask(self.x,self.y,self.a,self.b,self.theta,r,shape)

    def create_bisection_mask(self,shape):
        """Generate bisection masks (pos_mask,neg_mask)
        Ellipse has center (x,y), and angle theta

        Args:
            shape: The shape of the ellipse mask
        Returns:
            (pos_mask,neg_mask)
        """
        if not self.is_valid: 
            raise ValueError("Gofher Parameters (x={},y={},a={},b={}) must be positive values to create a bisection mask".format(self.x,self.y,self.a,self.b))
        if not isinstance(shape, tuple) or not len(shape) == 2 or not isinstance(shape[0],int) or not isinstance(shape[1],int) or min(shape) <= 0:
            raise ValueError("shape ({}) must be a tuple of 2 positive integers to create an ellipse mask ".format(shape)) 
        return create_bisection_mask(self.x,self.y,self.theta,shape)

    