function A = myQR(A)
% my QR factorization via Householder transformations

n = size(A,1);

for i = 1:(n-1)
    a = A(i:n,i);
    v = [zeros(i-1,1); a];
    v(i) = v(i) + sign(a(1)) * norm(a);
    
    % H = I - 2 * v * v' / (v' * v)
    % Instead of creating/applying H, apply it individually to each column of A
    
    v2 = 2/(v'*v) * v;
    for j = 1:n
        A(:,j) = A(:,j) - (v'*A(:,j)) * v2;
    end
end