% BIN2AVI
%
% Create avi-movie from single frames

clear;

% The list of images/Frames
fpath = '';                       % Frames path
%unzip('Long_movie.zip');
imglist = dir([fpath 'myFile*.bin']);% Frames name

width = 1024;                       % Frame width
height = 768;                       % Frame height
% Frame colormap:    color(i,j) = cmap( im(i,j) )
% Not needed (no effect) if image is truecolor (widthxheightx3) image
%cmap = bone(256);

% For all the frames
for k = 1:length(imglist);
  
    framename = [fpath imglist(k).name]          % Get current frame
    file = fopen(framename);
    img = fread(file,3*width*height,'uchar')/126;% Read the image (format dependent)
    fclose(file);
  
   thisFrame = reshape(img,3,width,height);
   thisFrame = permute(thisFrame,[3 2 1]);
   thisFrame = flipdim(thisFrame,1);

    %Potential Speedup
    %thisFrame = zeros(height,width,3);
    %thisFrame(:) = img(idxMap);
  
    % Save and png
    imwrite(thisFrame, strrep(framename,'.bin','.png'));
    % Draw to screen
    image(thisFrame); drawnow;
  
    % Insert as Frame
    frames(k) = im2frame(thisFrame); % Get the frame
end

% Create AVI
fps = 15;                         % frames per second
filename = 'myMovie.avi';               % path/name of movie output file
compress = 'None';                % Unix does not support AVI compression
%compress = 'Indeo3', 'Indeo5', 'Cinepak', 'MSVC', or 'None'   % Windows
movie2avi(frames, filename, 'compression', compress, 'fps', fps);

%delete('*.bin'); 

% play result:
%movie(m, 1, fps)