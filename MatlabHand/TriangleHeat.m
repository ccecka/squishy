%--------------------------------------------------------------------------
%   Function                            TriangleHeat 
%   Referred to in function             Assembly
%   Purpose                             Set up the element stiffness and vector 
%--------------------------------------------------------------------------

function [k_e, f_e, f_g] = TriangleHeat(nDim, nDoF, nEquations, nElements, Coord,...
                    ID, IEN, LM, BCVal, nNodesElement, f, h, Ce, ielem )

elementSize = ElementJacobian(ielem, Coord, IEN, nDim, nNodesElement); %Warning: returns j/2
   
k_e = TriaStiffHeat(nDim, nDoF, elementSize, Ce, ielem, Coord, IEN, nNodesElement);
   
f_e = TriaForceHeat(f, elementSize, nDim, nDoF);
   
f_g = TriaForceHeat_g(k_e, BCVal, IEN, ielem, nNodesElement, nDoF);

%--------------------------------------------------------------------------
%   Function                            TriaStiffHeat 
%   Referred to in function             TriangleHeat
%   Purpose                             Set up the element stiffness 
%--------------------------------------------------------------------------

function [k_e] = TriaStiffHeat(nDim, nDoF, elementSize, Ce, ielem, Coord, ...
                 IEN, nNodesElement)
%	Calculates the element stiffness matrix
B_e = [Coord(2,IEN([2,3,1],ielem)) - Coord(2,IEN([3,1,2],ielem));
       Coord(1,IEN([3,1,2],ielem)) - Coord(1,IEN([2,3,1],ielem))];
k_e = Ce/(4*elementSize) * B_e' * B_e;

%--------------------------------------------------------------------------
%   Function                            TriaForceHeat 
%   Referred to in function             TriangleHeat
%   Purpose                             Set up the element vector
%--------------------------------------------------------------------------

function [f_e] = TriaForceHeat(f, elementSize, nDim, nDoF)
%	Calulates the element force vector 
f_e = (f*elementSize/3) * ones(3*nDoF,1);

%--------------------------------------------------------------------------
%   Function                            TriaForceHeat_g 
%   Referred to in function             TriangleHeat
%   Purpose                             Set up the element vector
%--------------------------------------------------------------------------

function f_g = TriaForceHeat_g(k_e, BCVal, IEN, elementNumber, nNodesElement, nDoF)
% 	Calculates the element force vector due to the essential boundary
% 	condition.
f_g = -k_e * BCVal(IEN(:,elementNumber));

%--------------------------------------------------------------------------
%   Function                            ElementJacobian 
%   Referred to in function             Various element functions
%   Purpose                             Computes Jacobian (returns 1/2 J)
%--------------------------------------------------------------------------
                                        % For two dimensional meshes only
function [elementSize] = ElementJacobian(ielem, Coord, IEN, nDim, nNodesElement)

elementCoord = zeros(nDim,nNodesElement);
for i = 1:nDim
    for j = 1:nNodesElement
        globalNode = IEN(j,ielem);
        elementCoord(i,j) = Coord(i,globalNode);
    end
end

j_e = [elementCoord(1, 1) - elementCoord(1, 3),... 
       elementCoord(2, 1) - elementCoord(2, 3);
       elementCoord(1, 2) - elementCoord(1, 3),... 
       elementCoord(2, 2) - elementCoord(2, 3)];
elementSize = 0.5*det(j_e);

%--------------------------------------------------------------------------
%   End of file TriangleHeat.m
%--------------------------------------------------------------------------
