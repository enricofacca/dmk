

def test_case(label,ndiv):
    """
    Shorthand for definition of test case of optimal transport Problem.
    Args:
    label(str) :: defines the problem consider
    ndiv(int) :: 1/ndiv is proportional to the mesh size
    
    Returns:
    mesh: Spatial information of the problem
    forcing: function f:=f^{+}-f^{-}
    opttdens: optimal transport density 
    optpot: Kantorovich potential
    """

    if label == 'rect_uniform':
        """Define Optimal Transport density assocaited to 
        forcing term in readable in fenics assembler 
        Test case in 
        - Facca et. al 2018 https://doi.org/10.1137/16M1098383 
        - Facca et. a. 2021
        
        """
 
        # define mesh
        mesh = UnitSquareMesh(ndiv, ndiv)

        # define forcing term = f = f^{+}-f^{-} 
        x, y = SpatialCoordinate(mesh)
        source  = conditional( And(abs(x-0.25) < 0.125, abs(y-0.5)<0.25), 2.0, 0.0) 
        sink    = conditional( And(abs(x-0.75) < 0.125, abs(y-0.5)<0.25), 2.0, 0.0) 
        forcing = source - sink

        

    
    return mesh, forcing, opttdens, optpot
