load -ASCII K.mat;
K(:,1:2) = K(:,1:2) + 1;
K(:,3) = 1;
K = spconvert(K);
spy(K);