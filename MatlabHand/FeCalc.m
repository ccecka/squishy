%--------------------------------------------------------------------------
%   Matlab function FECALC
%--------------------------------------------------------------------------
%
%   This is a driver program to solve PDE's, using the Finite Element Method.
%
%   Original Author: Jee Rim 
%   Adapted by: James Tuck-Lee and Dolf van der Heide
%   ME335A Finite Element Methods, Winter 2007
%   Stanford University 
%
%   Some Important Variables:
%		nDim  : dimensionality of physical domain (i.e. 1D, 2D, 3D)
%     nDoF  : number of degrees of freedom per node
%     nNodes: number of nodes in the mesh
%     nNodesElement : number of nodes per element
%		nElements : number of elements in the mesh
%		elementType : which problem/element type to use (string)
%     Coord : the ndim by nNodes array of nodal coordinates	
%		f		: distributed load (value; vector for nDoF > 1)
%		h		: the array of Dirichlet boundary conditions (values applied to nNodes x nDoF)
%     BCVal : the array of essential boundary condition values (size = nNodes x nDoF)
%     BCIndex : the essential boundary condition index array (size = nNodes x nDoF)
%     ID  : the nNodes-by-nDoF array of global equation numbers 
%     IEN : the element nodes array (returns the global node number of a node)
%     LM  : the location matrix (returns the equation number corresponding to a node)
%     nEquations : number of equations for the problem
%     elementSize : the length of a linear element 
%		Ce  : material constant for elements
%     k_e : the element stiffness matrix
%     f_e : the element force vector due to a distributed load
%     f_g : the element force vector due to essential b.c.'s
%     K : the nEquations-by-nEquations global stiffness matrix	
%     F : the nEquations-by-1 global force vector
%     d : the degrees of freedom vector (nEquations-by-1)
%		disp : the solution (including essential b.c.'s)
%
%--------------------------------------------------------------------------
% Phase 1: Problem Definition
%--------------------------------------------------------------------------
clear;

[nDim, nDoF, nNodes, nElements, nNodesElement, Coord, CoordRef, nEquations, ...
    f, h, BCVal, BCIndex, IEN, LM, ID] = ProblemDefinition();


[F, K, M] = Assembly(nDim, nDoF, nNodes, nEquations, nElements, Coord, CoordRef, ID, IEN, LM, BCVal, nNodesElement, f, h);

Coord = Coord * 6/8;
tetramesh( IEN', Coord, ones(1,size(IEN,2)) );
axis(2*[-1,1,-1,1,-1,1]);
axis vis3d;
myAxis = axis;
drawnow;

dt = 0.01;

p = 0;

for k = 0:100000
    
    % Spin!
%     if( k == 250 )
%         f(4,2) = 50;
%         f(5,2) = -50;
%         f(6,1) = -50;
%         f(7,1) = 50;
%     elseif( k == 500 )
%         f(2,1) = 20;
%         f(3,1) = -20;
%     else
%         f(:) = 0;
%     end
        
    % Optimized Single-Iteration Newton-Raphson
    %[F, K] = Assembly(nDim, nDoF, nNodes, nEquations, nElements, Coord, CoordRef, ID, IEN, LM, BCVal, nNodesElement, f, h);
    
    %DH = (1/dt) * M + (dt/2) * K;

    %dx = DH \ (p - (dt/2) * F);
    %dCoord = ArrangeResults(dx, nNodes, nDoF, ID, BCVal, BCIndex);
    %dCoord = reshape(dx, 3, nNodes)';
    
    %[F] = Assembly(nDim, nDoF, nNodes, nEquations, nElements, Coord + 0.5*dCoord, CoordRef, ID, IEN, LM, BCVal, nNodesElement, f, h);
    
    %Coord = Coord + dCoord;
    %p = ((1/dt) * M * dx - (dt/2) * F);
    
    %k
    
    % Full Newton-Raphson method
    CoordK = Coord;
    
    Hp = 1;
    normHp = 1;
    NRiters = 0;
    
    while( normHp > 1e-13 )
        
      [F, K] = Assembly(nDim, nDoF, nNodes, nEquations, nElements, 0.5*(Coord+CoordK), CoordRef, ID, IEN, LM, BCVal, nNodesElement, f, h);
    
      DH = (1/dt) * M + (dt/2) * K;

      vp = (Coord - CoordK)' / dt;
      Hp = (p - M * vp(:) - (dt/2) * F);
      dx = DH \ Hp;
      dCoord = ArrangeResults(dx, nNodes, nDoF, ID, BCVal, BCIndex);

      Coord = Coord + dCoord;

      %tetramesh( IEN', Coord ); axis( myAxis ); drawnow;

      NRiters = NRiters + 1;
      
      normHp = norm( Hp );
    end
    
    [k, NRiters, norm(Hp)]

    [F] = Assembly(nDim, nDoF, nNodes, nEquations, nElements, 0.5*(Coord+CoordK), CoordRef, ID, IEN, LM, BCVal, nNodesElement, f, h);
    
    v = (Coord - CoordK)' / dt;
    p = (M * v(:) - (dt/2) * F);

    tetramesh( IEN', Coord, ones(1,size(IEN,2)) );
    axis( myAxis ); 
    drawnow;
    
end

%--------------------------------------------------------------------------
% Phase 4a: Results
%--------------------------------------------------------------------------

                                    % arrange the results to include the 
                                    % essential boundary conditions.
%[disp] = ArrangeResults(d, nNodes, nDoF, ID, BCVal, BCIndex);



%--------------------------------------------------------------------------
% Phase 5: Clean-up
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%   End of file FeCalc.m
%--------------------------------------------------------------------------