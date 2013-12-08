%--------------------------------------------------------------------------
%   Function                            ModelProb 
%   Referred to in function             Assembly
%   Purpose                             Set up the element stiffness and vector 
%--------------------------------------------------------------------------

function [k_e, f_e, f_g] = ModelProb(nDim, nDoF, nEquations, nElements, Coord,...
                    ID, IEN, LM, BCVal, nNodesElement, f, h, ielem )
    
elementSize = ModelProb_Size(ielem, Coord, IEN, nDim, nNodesElement);
   
k_e = ModelProb_Stiff(nDim, nDoF, elementSize, ielem, Coord, IEN, nNodesElement);
   
f_e = ModelProb_Force(f, elementSize, nDim, nDoF);
   
f_g = ModelProb_Force_g(k_e, BCVal, IEN, ielem, nNodesElement, nDoF);

%--------------------------------------------------------------------------
%   Function                            ModelProb_Size 
%   Referred to in function             ModelProb
%   Purpose                             Computes element size from Jacobian 
%--------------------------------------------------------------------------

function elementSize = ModelProb_Size(ielem, Coord, IEN, nDim, nNodesElement)

elemtCoord = zeros(nDim,nNodesElement);
for i = 1:nDim
    for j = 1:nNodesElement
        globalNode = IEN(j,ielem);
        elementCoord(i,j) = Coord(i,globalNode);
    end
end
j_e = elementCoord(1,2) - elementCoord(1,1);
elementSize = abs(j_e);

%--------------------------------------------------------------------------
%   Function                            ModelProb_Stiff
%   Referred to in function             ModelProb
%   Purpose                             Set up the element stiffness 
%--------------------------------------------------------------------------

function [k_e] = ModelProb_Stiff(nDim, nDoF, elementSize, ielem, Coord, ...
                 IEN, nNodesElement)
	
%	Calculates the element stiffness matrix of a 1-d modelproblem element.
%   Note: More input variables than necessary are listed for this simple case
%	The output [k_e] is the 2-by-2 element stiffness matrix.

k_e = 1/elementSize * [ 1 -1 ; -1 1 ];

%--------------------------------------------------------------------------
%   Function                            ModelProb_Force
%   Referred to in function             ModelProb
%   Purpose                             Set up the element force vector
%-------------------------------------------------------------------------- 

function [f_e] = ModelProb_Force(f, elementSize, nDim, nDoF)
																		
%	 Calulates the element force vector acting on a 1-d bar element due to
%	a constant distributed load f(x) = f.
%	 Takes as input arguments the constant force f and the length of the
%	element elementLength.
%	Outputs a nNodesElement*nDoF-by-1 force vector [f_e].

f_e = f*elementSize/2 * [ 1 ; 1 ];

%--------------------------------------------------------------------------
%   Function                            ModelProb_Force_g
%   Referred to in function             ModelProb
%   Purpose                             Set up the element force vector due
%                                       to the essential bc.
%--------------------------------------------------------------------------   

function f_g = ModelProb_Force_g(k_e, BCVal, IEN, elementNumber, nNodesElement, nDoF)

%   Calculates the element force vector due to the essential boundary
%  condition.
%   The input arguments are the element stiffness matrix k_e, the boundary 
%  value array BCVal and the element nodes array IEN, as well as the 
%  element number elementNumber and the number of nodes per element
%  nNodesElement.
%   The output is the nNodesElement-by-1 element force array due to the essential
%  b.c.'s

f_g = - k_e * BCVal(IEN(:, elementNumber));

%--------------------------------------------------------------------------
%   End of file ModelProb.m
%--------------------------------------------------------------------------