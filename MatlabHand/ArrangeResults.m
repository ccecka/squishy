%--------------------------------------------------------------------------
%   Function                            ArrangeResults
%   Referred to in function             FeCalc
%   Purpose                             Arrange results for display
%--------------------------------------------------------------------------

function disp = ArrangeResults(d, nNodes, nDoF, ID, BCVal, BCIndex)

disp = d(ID);

% disp = zeros(nNodes, nDoF);
% 
% for i = 1:nNodes
%     for j = 1:nDoF
%         if(BCIndex(i,j) == 0)
%             disp(i,j) = d(ID(i,j));
%         else
%             disp(i,j) = BCVal(i,j);
%         end
%     end
% end

%--------------------------------------------------------------------------
%   End of file ArrangeResults.m
%--------------------------------------------------------------------------
