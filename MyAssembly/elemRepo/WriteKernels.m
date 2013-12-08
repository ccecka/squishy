clear;

XYZ = ['x' 'y' 'z'];
int2xyz = @(k) XYZ(k);


FORCE = 0;
ELAST = 1;

CPU = 0;
GPU = 1;

REAL = 'T ';

for type = [FORCE ELAST]
    for host = [CPU GPU]

        KEfile = fopen('kernelMapK_e.ker', 'w');
        % include header code
        fprintf(KEfile,['#ifndef KERNELMAPK_E\n#define KERNELMAPK_E\n\ninline int kernelMapK_e( int k1, int k2 )\n{\n']);

        FEfile = fopen('kernelMapF_e.ker', 'w');
        % include header code
        fprintf(FEfile,['#ifndef KERNELMAPF_E\n#define KERNELMAPF_E\n\ninline int kernelMapF_e( int k1 )\n{\n']);
        
        switch( type )
            case FORCE
                filename = ['elastF'];
            case ELAST
                filename = ['elastKF'];
        end
        switch( host )
            case CPU
                filename = [filename '_CPU.ker'];
            case GPU
                filename = [filename '_GPU.ker'];
        end
        file = fopen(filename,'w');

        % Assume there already exists world coordinates
        % x1 x2 x3 x4
        % y1 y2 y3 y4
        % z1 z2 z3 z4
        % and material (upper triangular) inverse Jacobian (constant)
        % Jinv11 Jinv12 Jinv13
        %        Jinv22 Jinv23
        %               Jinv33
        % and external forces
        % bx1 bx2 bx3 bx4
        % by1 by2 by3 by4
        % bz1 bz2 bz3 bz4

        % Compute the extra shape function, volume, and PKc
        fprintf(file,['const ' REAL 'Jinv41 = -Jinv11;\n']);   % Can remove and replace with -Jinv11
        fprintf(file,['const ' REAL 'Jinv42 = -Jinv12-Jinv22;\n']);
        fprintf(file,['const ' REAL 'Jinv43 = -Jinv13-Jinv23-Jinv33;\n']);

        fprintf(file,['const ' REAL 'V = (ORIENT/6.0f)/(Jinv11*Jinv22*Jinv33);\n']);
        % Including the DT/2 in V for computing (DT/2)*F and (DT/2)*K
        %fprintf(file,['const ' REAL 'V = ((DT/2.0f)*(ORIENT/6.0f))/(Jinv11*Jinv22*Jinv33);\n']);
        switch( host )
            case CPU
                fprintf(file,['assert(V > 0);\n']);
            case GPU
                fprintf(file,['\n']);
        end

        fprintf(file,['\n']);

        % Compute F
        for i = 1:3
            for j = 1:3
                fprintf(file,[REAL 'F' int2str(i) int2str(j) ' = ']);
                for k = 1:j    % Account for upper tri Jinv
                    %fprintf(file,['Ds' int2str(i) int2str(k) '*Jinv' int2str(k) int2str(j) '+']);  
                    fprintf(file,['(' int2xyz(i) int2str(k) '-' int2xyz(i) '4)*Jinv' int2str(k) int2str(j) '+']);  
                end
                fseek(file,-1,0);  % Delete last ' +'
                fprintf(file,[';\n']);
            end
        end

        fprintf(file,['\n']);

        % Compute FinvT (Optimized)
        fprintf(file,[REAL 'FinvT11 = F22*F33-F32*F23;\n']);
        fprintf(file,[REAL 'FinvT12 = F23*F31-F21*F33;\n']);
        fprintf(file,[REAL 'FinvT13 = F21*F32-F22*F31;\n']);
        fprintf(file,[REAL 'FinvT33 = F11*FinvT11+F12*FinvT12+F13*FinvT13;    // Temp Det\n']);
        switch( host )
            case CPU
                fprintf(file,['assert( FinvT33 > 0 );\n']);
            case GPU
                fprintf(file,['\n']);
        end
        fprintf(file,[REAL 'PKc = BULK_MOD*logf(FinvT33) - SHEAR_MOD;\n']); 
        fprintf(file,['FinvT33 = 1.0f/FinvT33;\n']);
        fprintf(file,['FinvT11 *= FinvT33;\n']);
        fprintf(file,['FinvT12 *= FinvT33;\n']);
        fprintf(file,['FinvT13 *= FinvT33;\n']);
        fprintf(file,[REAL 'FinvT21 = (F13*F32-F12*F33)*FinvT33;\n']);
        fprintf(file,[REAL 'FinvT22 = (F11*F33-F13*F31)*FinvT33;\n']);
        fprintf(file,[REAL 'FinvT23 = (F12*F31-F11*F32)*FinvT33;\n']);
        fprintf(file,[REAL 'FinvT31 = (F12*F23-F13*F22)*FinvT33;\n']);
        fprintf(file,[REAL 'FinvT32 = (F13*F21-F11*F23)*FinvT33;\n']);
        fprintf(file,['FinvT33 = (F11*F22-F12*F21)*FinvT33;\n']);

        fprintf(file,['\n']);

        % Compute k_e and f_e

        % Declare a body force
        fprintf(file,[REAL 'BodyF;\n']);
        
        counter = 0;

        % Compute f_e
        for i = 1:3
            % Compute P(i,J) (reuse Fij space)
            for J = 1:3
                fprintf(file,['F' int2str(i) int2str(J) ' = SHEAR_MOD*F' int2str(i) int2str(J) '+PKc*FinvT' int2str(i) int2str(J) ';\n']);
            end

            % Compute External Body Force
            fprintf(file,['BodyF = -0.05f*(b' int2xyz(i) '1+b' int2xyz(i) '2+b' int2xyz(i) '3+b' int2xyz(i) '4);\n']);

            % Compute f_e
            for a = 1:4
                if( host == CPU )
                    fprintf(file,['f_e[' int2str(3*(a-1)+i-1) ']']);
                else
                    fprintf(file,['E[' int2str(counter) '*ESTRIDE]']);
                    % Write to the FEMap so we don't have to compute it...
                    fprintf(FEfile,['  if( k1 == ' int2str(3*(a-1)+i-1) ' ) return ' int2str(counter) ';\n']);
                    counter = counter + 1;
                end
                fprintf(file,[' = V*(']);

                % Internal forces
                for J = 1:3
                    if( a <= J | a == 4 )
                        fprintf(file,['F' int2str(i) int2str(J) '*Jinv' int2str(a) int2str(J) '+']);
                    end
                end

                % Add External Body Force and External Node Force
                fprintf(file,['BodyF-0.05f*b' int2xyz(i) int2str(a)]);
                % Lumped force?
                %fprintf(file,['-0.25f*b' int2xyz(i) int2str(a)]);
                fprintf(file,[');\n']);
            end
        end

        fprintf(file,['\n']);

        if( type == FORCE )
            fclose(file);
            continue;
        end

        % Compute k_e
        for i = 1:3
            for j = 1:3
                % Compute DP(i,J,j,K) (reuse Fij space)
                for J = 1:3
                    for K = 1:3
                        fprintf(file,['F' int2str(J) int2str(K) ' = ']);
                        if( i == j && J == K )
                            fprintf(file,['SHEAR_MOD+']);
                        end
                        fprintf(file,['BULK_MOD*FinvT' int2str(i) int2str(J) '*FinvT' int2str(j) int2str(K) '-PKc*FinvT' int2str(K) int2str(i) '*FinvT' int2str(J) int2str(j) ';\n']);
                    end
                end

                % Compute k_e
                for a = 1:4
                    for b = 1:4
                        if( 3*(a-1)+i <= 3*(b-1)+j )    % k_e is symm
                           	
                            if( host == CPU )
                                fprintf(file,['k_e(' int2str(3*(a-1)+i-1) ',' int2str(3*(b-1)+j-1) ')']);
                            else
                                fprintf(file,['E[' int2str(counter) '*ESTRIDE]']);
                                % Write to the KEMap so we don't have to compute it...
                                fprintf(KEfile,['  if( k1 == ' int2str(3*(a-1)+i-1) ' && k2 == ' int2str(3*(b-1)+j-1) ' ) return ' int2str(counter) ';\n']);
                                % And the transpose
                                fprintf(KEfile,['  if( k1 == ' int2str(3*(b-1)+j-1) ' && k2 == ' int2str(3*(a-1)+i-1) ' ) return ' int2str(counter) ';\n']);
                                counter = counter + 1;
                            end
                            
                            fprintf(file,[' = V*(']);
                            % Mass matrix contribution
                            %if( i == j )
                                %if( a == b )
                                    % (2/DT) * (1/DT) * MASS / 4
                                    %fprintf(file,['(MASS/(DT*DT*2.0f))+']);
                                    % (1/DT) * MASS / 4
                                    %fprintf(file,['(MASS/(DT*4.0f))+']);
                                %end
                            %end
                            for J = 1:3
                                for K = 1:3
                                    if( (a <= J | a == 4) & (b <= K | b == 4) )
                                        fprintf(file,['Jinv' int2str(a) int2str(J) '*F' int2str(J) int2str(K) '*Jinv' int2str(b) int2str(K) '+']);
                                    end
                                end
                            end
                            fseek(file,-1,0);  % Delete last ' +'
                            fprintf(file,[');\n']);
                        end
                    end
                end
            end
        end

        fclose(file);
        
        % Finish and close
        fprintf(FEfile,['return -1;\n}\n\n#endif\n\n']);
        fclose(FEfile);
        fprintf(KEfile,['return -1;\n}\n\n#endif\n\n']);
        fclose(KEfile);

    end
end


