%--------------------------------------------------------------------------
%   Function                            TriangleHeat 
%   Referred to in function             Assembly
%   Purpose                             Set up the element stiffness and vector 
%--------------------------------------------------------------------------

function [f_e, k_e, m_e] = TetElastMKF(nDim, nDoF, nEquations, nElements, Coord, CoordRef, ...
                                            ID, IEN, LM, BCVal, nNodesElement, f, h, ielem )

mu = 0.017;
lambda = 0.155;
%mu = 1;
%lambda = 1;
mass = 1;

% Derivative of parent shape functions  dN_i / dE_j
H = [ 1  0  0
      0  1  0
      0  0  1
     -1 -1 -1 ];

Ds = Coord(IEN([1,2,3,4],ielem),:)' * H;
Dm = CoordRef(IEN([1,2,3,4],ielem),:)' * H;
[QDm,RDm] = qr(Dm);
if( det(QDm) < 0 )
    RDm(3,3) = -RDm(3,3);
end

% Jacobian    dX_i/dE_j
%J = RDm;
J = Dm
V = det(J)/6;
%V = prod(diag(J))/6
assert(V > 0)

% Compute the inverse Jacobian   Jinv_{ij} = dE_i/dX_j
Jinv = inv(J);

F = Ds*Jinv;
assert( det(F) > 0 )
FinvT = inv(F)';

% Transform to element           dN_i/dX_j = (dN_i/dE_k) * (dE_k/dX_j)
Grad = Jinv;  
Grad(4,:) = - Grad(1,:) - Grad(2,:) - Grad(3,:);

PKc = lambda*log(det(F)) - mu;
% P = mu * F + PKc * FinvT;

I = eye(3);

% DP(3,3,3,3) = 0;
% for i = 1:3
%     for j = 1:3
%         for k = 1:3
%             for l = 1:3
%                 DP(i,j,k,l) = mu*I(i,k)*I(j,l) + lambda*FinvT(i,j)*FinvT(k,l) - PKc*FinvT(l,i)*FinvT(j,k);
%             end
%         end
%     end
% end

k_e(12,12) = 0;
f_e(12) = 0;
%m_e = diag(mass * V/4 * ones(12,1));
% for a = 1:4
%     for i = 1:3
%         for b = 1:4
%             for j = 1:3
%                 kij = 0;
%                 for J = 1:3
%                     for K = 1:3
%                         kij = kij + Grad(a,J) * DP(i,J,j,K) * Grad(b,K);
%                     end
%                 end
%                 k_e((a-1)*3+i,(b-1)*3+j) = V * kij;
%                 % m_e((a-1)*3+i,(b-1)*3+j) = mass * I(i,j) * V * 1/(20 - 10*(a == b));
%             end
%         end
%         f_e((a-1)*3+i) = V * sum( P(i,:) .* Grad(a,:) );
%     end
% end

for i = 1:3
    for j = 1:3
        % Compute DP(i,J,j,K)
        for J = 1:3
            for K = 1:3
                DP(J,K) = mu*I(i,j)*I(J,K) + lambda*FinvT(i,J)*FinvT(j,K) - PKc*FinvT(K,i)*FinvT(J,j);
            end
        end
        
        k_e((0:3:9)+i,(0:3:9)+j) = V * (Grad * DP * Grad.');
        
        % Compute P(i,J)
        Pi = mu * F(i,:) + PKc * FinvT(i,:);
        f_e((0:3:9)+i) = V * (Pi * Grad.');
    end
end


% Check f_e computation
% For the first node in material coordinates, compute normal vectors and
% areas of adjacent faces
% r2 = CoordRef(IEN(2,ielem),:) - CoordRef(IEN(1,ielem),:);
% r3 = CoordRef(IEN(3,ielem),:) - CoordRef(IEN(1,ielem),:);
% r4 = CoordRef(IEN(4,ielem),:) - CoordRef(IEN(1,ielem),:);
% An23 = cross( r2, r3 )/2;
% An34 = cross( r3, r4 )/2;
% An42 = cross( r4, r2 )/2;
% 
% f_g(1:3) = -P * (An23 + An34 + An42)'/3;
% 
% r2 = CoordRef(IEN(1,ielem),:) - CoordRef(IEN(2,ielem),:);
% r3 = CoordRef(IEN(4,ielem),:) - CoordRef(IEN(2,ielem),:);
% r4 = CoordRef(IEN(3,ielem),:) - CoordRef(IEN(2,ielem),:);
% An23 = cross( r2, r3 )/2;
% An34 = cross( r3, r4 )/2;
% An42 = cross( r4, r2 )/2;
% 
% f_g(4:6) = -P * (An23 + An34 + An42)'/3;
% 
% r2 = CoordRef(IEN(1,ielem),:) - CoordRef(IEN(3,ielem),:);
% r3 = CoordRef(IEN(2,ielem),:) - CoordRef(IEN(3,ielem),:);
% r4 = CoordRef(IEN(4,ielem),:) - CoordRef(IEN(3,ielem),:);
% An23 = cross( r2, r3 )/2;
% An34 = cross( r3, r4 )/2;
% An42 = cross( r4, r2 )/2;
% 
% f_g(7:9) = -P * (An23 + An34 + An42)'/3;
% 
% r2 = CoordRef(IEN(1,ielem),:) - CoordRef(IEN(4,ielem),:);
% r3 = CoordRef(IEN(3,ielem),:) - CoordRef(IEN(4,ielem),:);
% r4 = CoordRef(IEN(2,ielem),:) - CoordRef(IEN(4,ielem),:);
% An23 = cross( r2, r3 )/2;
% An34 = cross( r3, r4 )/2;
% An42 = cross( r4, r2 )/2;
% 
% f_g(10:12) = -P * (An23 + An34 + An42)'/3;
% 
% [f_e', f_g']
% 
% f_e = -f_g;


% Assemble Body Forces
N = [1 0 0 1 0 0 1 0 0 1 0 0; 0 1 0 0 1 0 0 1 0 0 1 0; 0 0 1 0 0 1 0 0 1 0 0 1];
b = f(IEN([1,2,3,4],ielem),:)';
b = b(:);

f_e = f_e' + (V/20) * (N' * N * b + b);
m_e = mass * (V/20) * (N' * N + eye(12));

f_e

%--------------------------------------------------------------------------
%   End of file TetElast.m
%--------------------------------------------------------------------------
