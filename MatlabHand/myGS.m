function [Q,R] = myGS(A)
% Function to compute the QR of A via Gramm Schmidt

n = size(A,1);
Q = A;

for j = 1:n
    for k = 1:j-1
        Q(:,j) = Q(:,j) - dot(Q(:,k),A(:,j)) * Q(:,k);
    end
    Q(:,j) = Q(:,j) / norm(Q(:,j));
end

R = Q'*A;